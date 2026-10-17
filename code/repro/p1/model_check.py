"""Print layer / parameter counts (unfused and fused) for P1 model yamls.

Builds each model with the GBH-YOLOv5 repo's own `models/yolo.py` so counts match
what its train.py / test.py print. Table 6 of P1 is measured "on the test set",
where test.py fuses Conv+BN first, so the fused numbers are the comparable ones.

usage: python model_check.py --repo code/gbh-yolov5 cfg1.yaml [cfg2.yaml ...]
"""
import argparse
import os
import sys
from copy import deepcopy

ap = argparse.ArgumentParser()
ap.add_argument('--repo', required=True)
ap.add_argument('--img', type=int, default=960)
ap.add_argument('cfgs', nargs='+')
a = ap.parse_args()

cfgs = [os.path.abspath(c) for c in a.cfgs]
sys.path.insert(0, os.path.abspath(a.repo))
os.chdir(a.repo)

import torch  # noqa: E402
from models.yolo import Model  # noqa: E402


def counts(m):
    return len(list(m.modules())), sum(p.numel() for p in m.parameters())


def gflops(m, img):
    try:
        from thop import profile
        x = torch.zeros(1, 3, img, img)
        return profile(deepcopy(m), inputs=(x,), verbose=False)[0] / 1e9 * 2
    except Exception:
        return float('nan')


print(f"{'cfg':40s} {'layers':>7s} {'params':>11s} | {'fused L':>7s} {'fused P':>11s} | GFLOPs@{a.img}")
for c in cfgs:
    m = Model(c, ch=3).eval()
    l0, p0 = counts(m)
    g = gflops(m, a.img)
    m.fuse()
    l1, p1 = counts(m)
    print(f"{os.path.basename(c):40s} {l0:7d} {p0:11,d} | {l1:7d} {p1:11,d} | {g:.1f}")
