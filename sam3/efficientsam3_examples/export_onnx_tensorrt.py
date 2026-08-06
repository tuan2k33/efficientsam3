#!/usr/bin/env python3
"""
Export an EfficientSAM3 image model to ONNX (and optionally a TensorRT engine)
for a fixed set of text classes, baked into the graph as constants.

Usage:
    python3 export_onnx_tensorrt.py --checkpoint path/to/efficientsam3_efficientvit.pt \\
        --backbone-type efficientvit --model-name b1 \\
        --text-encoder-type MobileCLIP-S0 --text-encoder-context-length 16 \\
        --classes person car bus --device cuda --build-engine
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
from sam3.model.data_misc import FindStage, interpolate
from sam3.model import box_ops

IMGSZ = 1008
NUM_TOP_QUERIES = 200


def encode_concepts(model, classes, device):
    print(f'[Encode] {len(classes)} class(es): {classes}')
    encoded = []
    for cls in classes:
        text_out = model.backbone.forward_text([cls], device=device)
        encoded.append(text_out)
        print(f"  [+] '{cls}'  language_features {tuple(text_out['language_features'].shape)}")
    return encoded


class EfficientSam3SpecializedWrapper(nn.Module):
    """
    Bakes a fixed list of text classes into the graph as buffers (become ONNX
    constants once exported), so the exported model takes only `pixel_values`.

    with_mask=False (default):
        pixel_values [B,3,H,W]  ->  detections [B, N_cls*Q, 6]
        columns: x1 y1 x2 y2 score cls_id  (pixel space, score=sigmoid(logit)*presence)

    with_mask=True:
        pixel_values [B,3,H,W]  ->  detections [B, N_cls*Q, 6]
                                    masks      [B, N_cls*Q, Hm, Wm]
        cls_id for query i: i // Q  (Q=num_top_queries)
    """

    def __init__(self, model, classes, device, imgsz=IMGSZ, with_mask=False,
                 num_top_queries=NUM_TOP_QUERIES):
        super().__init__()
        self.model = model
        self.num_classes = len(classes)
        self.with_mask = with_mask
        self.num_top_queries = num_top_queries

        encoded = encode_concepts(model, classes, device)
        for i, text_out in enumerate(encoded):
            self.register_buffer(f'text_features_{i}', text_out['language_features'])
            self.register_buffer(f'text_mask_{i}', text_out['language_mask'])
            self.register_buffer(f'text_embeds_{i}', text_out['language_embeds'])

        self.find_stage = FindStage(
            img_ids=torch.tensor([0], device=device, dtype=torch.long),
            text_ids=torch.tensor([0], device=device, dtype=torch.long),
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None,
        )
        self.register_buffer('scale', torch.tensor([imgsz, imgsz, imgsz, imgsz], dtype=torch.float32))

    def forward(self, pixel_values: torch.Tensor):
        B = pixel_values.shape[0]
        backbone_out = self.model.backbone.forward_image(pixel_values)
        geometric_prompt = self.model._get_dummy_prompt()

        det_chunks, mask_chunks = [], []
        for i in range(self.num_classes):
            backbone_out = dict(backbone_out)
            # language_features/embeds are (Seq, Batch, Dim); language_mask is (Batch, Seq)
            backbone_out['language_features'] = getattr(self, f'text_features_{i}').expand(-1, B, -1)
            backbone_out['language_mask'] = getattr(self, f'text_mask_{i}').expand(B, -1)
            backbone_out['language_embeds'] = getattr(self, f'text_embeds_{i}').expand(-1, B, -1)

            outputs = self.model.forward_grounding(
                backbone_out=backbone_out,
                find_input=self.find_stage,
                geometric_prompt=geometric_prompt,
                find_target=None,
            )
            out_bbox = outputs['pred_boxes']
            out_logits = outputs['pred_logits']
            presence = outputs['presence_logit_dec'].sigmoid().unsqueeze(1)
            scores = (out_logits.sigmoid() * presence).squeeze(-1)  # [B, Q]

            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox) * self.scale
            cls_col = torch.full_like(scores, float(i)).unsqueeze(-1)
            det_chunks.append(torch.cat([boxes, scores.unsqueeze(-1), cls_col], dim=-1))

            if self.with_mask:
                m = interpolate(outputs['pred_masks'], pixel_values.shape[-2:],
                                 mode='bilinear', align_corners=False).sigmoid()
                mask_chunks.append(m)

        detections = torch.cat(det_chunks, dim=1).half()  # [B, N_cls*Q, 6]
        if self.with_mask:
            return detections, torch.cat(mask_chunks, dim=1).half()  # [B, N_cls*Q, Hm, Wm]
        return detections


def build_mixed_precision_engine(onnx_path, engine_path, imgsz, min_batch, opt_batch, max_batch,
                                  input_name='pixel_values', workspace_gb=8,
                                  force_fp32_match='context_module'):
    """FP16 engine with FP32 pinned for `context_module` layers (overflow FP16 otherwise)."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    print(f'[Parse] {onnx_path} ...')
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f'  {parser.get_error(i)}')
            raise RuntimeError('ONNX parse failed')

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    n_forced = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if force_fp32_match not in layer.name:
            continue
        # Skip shape/index/bool-producing layers -- only float/half tensors can be
        # reassigned to FP32; forcing int64/bool outputs to float breaks the graph.
        out_dtypes = [layer.get_output(k).dtype for k in range(layer.num_outputs)]
        if not all(dt in (trt.float32, trt.float16) for dt in out_dtypes):
            continue
        layer.precision = trt.float32
        for k in range(layer.num_outputs):
            layer.set_output_type(k, trt.float32)
        n_forced += 1
    print(f'[Precision] Forced {n_forced}/{network.num_layers} layers to FP32 '
          f'(matched "{force_fp32_match}")')

    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_name,
        (min_batch, 3, imgsz, imgsz),
        (opt_batch, 3, imgsz, imgsz),
        (max_batch, 3, imgsz, imgsz),
    )
    config.add_optimization_profile(profile)

    print(f'[Build] shapes min={min_batch} opt={opt_batch} max={max_batch} ...')
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('TensorRT engine build failed')

    with open(engine_path, 'wb') as f:
        f.write(serialized)
    print(f'[Build] Saved: {engine_path}  ({(time.perf_counter() - t0) / 60:.1f} min)')


def export_and_build(args):
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

    print(f'[Build] EfficientSam3SpecializedWrapper (with_mask={args.mask}) ...')
    wrapper = EfficientSam3SpecializedWrapper(
        model, args.classes, device, imgsz=args.imgsz, with_mask=args.mask,
    ).to(device).eval()

    dummy = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    print('[Check] Sanity forward pass ...')
    with torch.no_grad():
        out = wrapper(dummy)

    nc = len(args.classes)
    if args.mask:
        detections, masks = out
        Q = detections.shape[1] // nc
        print(f'[Check] detections {tuple(detections.shape)}  expect [1,{nc * Q},6]')
        print(f'[Check] masks      {tuple(masks.shape)}')
        output_names = ['detections', 'masks']
        dynamic_axes = {'pixel_values': {0: 'batch'}, 'detections': {0: 'batch'}, 'masks': {0: 'batch'}}
    else:
        detections = out
        Q = detections.shape[1] // nc
        print(f'[Check] detections {tuple(detections.shape)}  expect [1,{nc * Q},6]')
        output_names = ['detections']
        dynamic_axes = {'pixel_values': {0: 'batch'}, 'detections': {0: 'batch'}}

    print(f'[Check] boxes  [{detections[0, :, 0].min():.1f}, {detections[0, :, 2].max():.1f}]')
    print(f'[Check] scores [{detections[0, :, 4].min():.3f}, {detections[0, :, 4].max():.3f}]')

    print(f'[Export] {args.output} ...')
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, args.output,
            input_names=['pixel_values'],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            dynamo=False,
        )
    size_mb = os.path.getsize(args.output) / 1e6
    print(f'[Export] Saved: {args.output}  ({size_mb:.0f} MB)')

    if args.build_engine:
        engine_path = args.output.replace('.onnx', '.engine')
        build_mixed_precision_engine(
            args.output, engine_path, args.imgsz,
            args.min_batch, args.opt_batch, args.max_batch,
        )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--classes', nargs='+', required=True)
    p.add_argument('--backbone-type', default='efficientvit')
    p.add_argument('--model-name', default='b1')
    p.add_argument('--text-encoder-type', default='MobileCLIP-S0')
    p.add_argument('--text-encoder-context-length', type=int, default=16)
    p.add_argument('--imgsz', type=int, default=IMGSZ)
    p.add_argument('--opset', type=int, default=17)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--mask', action='store_true', help='also export instance masks')
    p.add_argument('--output', default='efficientsam3.onnx')
    p.add_argument('--build-engine', action='store_true', help='also build a TensorRT engine (requires tensorrt)')
    p.add_argument('--min-batch', type=int, default=1)
    p.add_argument('--opt-batch', type=int, default=4)
    p.add_argument('--max-batch', type=int, default=16)
    return p.parse_args()


if __name__ == '__main__':
    export_and_build(parse_args())
