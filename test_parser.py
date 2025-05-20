#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from compare_gfl_inference import InferenceLogParser

jittor_log = '/root/data-fs/GFocalV2/inference_results/gfl_jittor_20250520_133453/inference.log'
pytorch_log = '/root/data-fs/GFocalV2/inference_results/gfl_pytorch_20250520_130518/inference.log'

print('Testing Jittor log parsing...')
jittor_parser = InferenceLogParser(jittor_log)
jittor_metrics = jittor_parser.parse_jittor_log()
print(f'Jittor metrics:')
print(f'  Inference times: {len(jittor_metrics["inference_time"])} records')
print(f'  Average time: {sum(jittor_metrics["inference_time"])/len(jittor_metrics["inference_time"]):.4f} seconds')
print(f'  FPS: {jittor_metrics["fps"]:.2f}')
print(f'  mAP: {jittor_metrics["bbox_mAP"]}')
print(f'  mAP@50: {jittor_metrics["bbox_mAP_50"]}')

print('\nTesting PyTorch log parsing...')
pytorch_parser = InferenceLogParser(pytorch_log)
pytorch_metrics = pytorch_parser.parse_pytorch_log()
print(f'PyTorch metrics:')
print(f'  Inference times: {len(pytorch_metrics["inference_time"])} records')
if len(pytorch_metrics["inference_time"]) > 0:
    print(f'  Average time: {sum(pytorch_metrics["inference_time"])/len(pytorch_metrics["inference_time"]):.4f} seconds')
    print(f'  FPS: {pytorch_metrics["fps"]:.2f}')
print(f'  mAP: {pytorch_metrics["bbox_mAP"]}')
print(f'  mAP@50: {pytorch_metrics["bbox_mAP_50"]}') 