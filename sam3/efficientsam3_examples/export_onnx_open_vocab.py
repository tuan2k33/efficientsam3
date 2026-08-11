#!/usr/bin/env python3
"""
Export the full open-vocabulary EfficientSAM3 image model to ONNX as two graphs:

  1. Text encoder  : tokenized_ids [B, ctx_len] (int64) -> language_mask, language_features, language_embeds
  2. Vision+decoder: pixel_values [B,3,H,W] + the three language_* tensors above -> detections [B, Q, 6]

The BPE tokenizer stays in Python (it isn't tensor ops, so it can't be part of
the ONNX graph) -- run it once per prompt, feed the resulting token ids into
graph 1, then feed both graph 1's output and the image into graph 2. Unlike
the fixed-class "specialize" export, no text is baked in: any prompt can be
tokenized and encoded at runtime.

The input resolution is fixed at IMGSZ (1008), matching the resolution the
checkpoints are trained and evaluated at. It is deliberately not exposed as a
flag: `_create_position_encoding(precompute_resolution=1008)` in model_builder
prepopulates `PositionEmbeddingSine.cache` for the four 1008-derived feature
map sizes (252/126/63/31), so `forward()` returns a cached constant that the
tracer bakes into the graph. Exporting at another resolution therefore
produces a graph whose positional encoding silently belongs to a different
input size. Making the export resolution-agnostic needs that cache bypassed
during tracing (and the compute path verified to trace to dynamic shape ops)
-- out of scope here, tracked separately.

Usage:
    python3 export_onnx_open_vocab.py --checkpoint path/to/efficientsam3_efficientvit.pt \\
        --backbone-type efficientvit --model-name b1 \\
        --text-encoder-type MobileCLIP-S0 --text-encoder-context-length 16 \\
        --device cuda --output-dir onnx_open_vocab
"""

import os
import sys
import time
import argparse

import torch
import torch.nn as nn

workspace_root = os.path.dirname(os.path.abspath(__file__))
sam3_repo_root = os.path.dirname(workspace_root)
sys.path.insert(0, sam3_repo_root)

from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.data_misc import FindStage
from sam3.model import box_ops

IMGSZ = 1008


class TextEncoderWrapper(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.encoder = text_encoder.encoder
        self.projector = text_encoder.projector

    def forward(self, tokenized_ids: torch.Tensor):
        input_embeds = self.encoder.forward_embedding(tokenized_ids)
        text_memory = self.encoder(input_embeds, return_all_tokens=True, input_is_embeddings=True)
        text_memory = self.projector(text_memory)
        language_mask = (tokenized_ids != 0).bool().ne(1)
        return language_mask, text_memory.transpose(0, 1), input_embeds.transpose(0, 1)


class VisionDecoderWrapper(nn.Module):
    """pixel_values + precomputed language_* tensors -> detections [B, Q, 6]
    (x1, y1, x2, y2, score, presence-gated). No class id column: this graph
    handles one prompt per call; batch over prompts by calling it once per
    prompt (or once per batch element if every image shares the same prompt)."""

    def __init__(self, model, imgsz=IMGSZ):
        super().__init__()
        self.model = model
        self.register_buffer('scale', torch.tensor([imgsz, imgsz, imgsz, imgsz], dtype=torch.float32))

    def forward(self, pixel_values, language_features, language_mask, language_embeds):
        B = pixel_values.shape[0]
        backbone_out = self.model.backbone.forward_image(pixel_values)
        backbone_out['language_features'] = language_features
        backbone_out['language_mask'] = language_mask
        backbone_out['language_embeds'] = language_embeds

        find_stage = FindStage(
            img_ids=torch.arange(B, device=pixel_values.device, dtype=torch.long),
            text_ids=torch.zeros(B, device=pixel_values.device, dtype=torch.long),
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None,
        )
        geometric_prompt = self.model._get_dummy_prompt(num_prompts=B)

        outputs = self.model.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_stage,
            geometric_prompt=geometric_prompt,
            find_target=None,
        )
        out_bbox = outputs['pred_boxes']
        out_logits = outputs['pred_logits']
        presence = outputs['presence_logit_dec'].sigmoid().unsqueeze(1)
        scores = (out_logits.sigmoid() * presence).squeeze(-1)  # [B, Q]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox) * self.scale
        return torch.cat([boxes, scores.unsqueeze(-1)], dim=-1)  # [B, Q, 5]


def export(args):
    device = args.device
    t0 = time.perf_counter()
    model = build_efficientsam3_image_model(
        checkpoint_path=args.checkpoint,
        backbone_type=args.backbone_type,
        model_name=args.model_name,
        text_encoder_type=args.text_encoder_type,
        text_encoder_context_length=args.text_encoder_context_length,
        load_from_HF=False,
        device=device,
    )
    print(f'[Load] {time.perf_counter() - t0:.1f}s')
    os.makedirs(args.output_dir, exist_ok=True)

    ctx_len = args.text_encoder_context_length
    text_wrapper = TextEncoderWrapper(model.backbone.language_backbone).to(device).eval()
    dummy_ids = torch.randint(1, 1000, (1, ctx_len), device=device, dtype=torch.long)

    print('[Check] Text encoder sanity forward pass ...')
    with torch.no_grad():
        language_mask, language_features, language_embeds = text_wrapper(dummy_ids)
    print(f'[Check] language_features {tuple(language_features.shape)} '
          f'language_mask {tuple(language_mask.shape)} '
          f'language_embeds {tuple(language_embeds.shape)}')

    text_onnx_path = os.path.join(args.output_dir, 'text_encoder.onnx')
    print(f'[Export] {text_onnx_path} ...')
    with torch.no_grad():
        torch.onnx.export(
            text_wrapper, dummy_ids, text_onnx_path,
            input_names=['tokenized_ids'],
            output_names=['language_mask', 'language_features', 'language_embeds'],
            dynamic_axes={'tokenized_ids': {0: 'batch'},
                           'language_mask': {0: 'batch'},
                           'language_features': {1: 'batch'},
                           'language_embeds': {1: 'batch'}},
            opset_version=args.opset,
            dynamo=False,
        )
    print(f'[Export] Saved: {text_onnx_path}')

    vision_wrapper = VisionDecoderWrapper(model, imgsz=IMGSZ).to(device).eval()
    dummy_pixels = torch.randn(1, 3, IMGSZ, IMGSZ, device=device)

    print('[Check] Vision+decoder sanity forward pass ...')
    with torch.no_grad():
        detections = vision_wrapper(dummy_pixels, language_features, language_mask, language_embeds)
    print(f'[Check] detections {tuple(detections.shape)}')

    vision_onnx_path = os.path.join(args.output_dir, 'vision_decoder.onnx')
    print(f'[Export] {vision_onnx_path} ...')
    with torch.no_grad():
        torch.onnx.export(
            vision_wrapper,
            (dummy_pixels, language_features, language_mask, language_embeds),
            vision_onnx_path,
            input_names=['pixel_values', 'language_features', 'language_mask', 'language_embeds'],
            output_names=['detections'],
            dynamic_axes={'pixel_values': {0: 'batch'},
                           'language_features': {1: 'batch'},
                           'language_mask': {0: 'batch'},
                           'language_embeds': {1: 'batch'},
                           'detections': {0: 'batch'}},
            opset_version=args.opset,
            dynamo=False,
        )
    print(f'[Export] Saved: {vision_onnx_path}')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--backbone-type', default='efficientvit')
    p.add_argument('--model-name', default='b1')
    p.add_argument('--text-encoder-type', default='MobileCLIP-S0')
    p.add_argument('--text-encoder-context-length', type=int, default=16)
    p.add_argument('--opset', type=int, default=17)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output-dir', default='onnx_open_vocab')
    return p.parse_args()


if __name__ == '__main__':
    export(parse_args())
