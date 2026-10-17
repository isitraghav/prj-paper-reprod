"""Layers / params / GFLOPs@640 for YOLOv5 v7.0 model yamls, computed the way v7.0's model_info does.

usage (from code/yolov5-v7.0): python ../repro/p2/gflops.py --nc 6 cfg1.yaml [cfg2.yaml ...]
"""
import argparse
import os
import sys
from copy import deepcopy

sys.path.insert(0, os.getcwd())
import torch  # noqa: E402
import thop  # noqa: E402
from models.yolo import DetectionModel  # noqa: E402
from utils.general import LOGGER  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--nc', type=int, default=6)
ap.add_argument('--img', type=int, default=640)
ap.add_argument('cfgs', nargs='+')
a = ap.parse_args()
LOGGER.setLevel(40)
print(f"{'cfg':42s} {'layers':>6s} {'params':>11s} {'GFLOPs':>7s}")
for c in a.cfgs:
    m = DetectionModel(c, ch=3, nc=a.nc).eval()
    s = max(int(m.stride.max()), 32)
    f = thop.profile(deepcopy(m), inputs=(torch.empty(1, 3, s, s),), verbose=False)[0] / 1e9 * 2 * (a.img / s) ** 2
    print(f'{os.path.basename(c):42s} {len(list(m.modules())):6d} {sum(p.numel() for p in m.parameters()):11,d} {f:7.1f}')
