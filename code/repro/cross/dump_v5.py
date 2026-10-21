"""Cross-dataset inference for the YOLOv5-family detectors (P1: GBH fork @51ee042 / v5.0; P2: yolov5 v7.0 + patch).

Runs a trained model, unchanged, on an image list and writes every detection after NMS, in original-image pixels:
  image,cls,conf,x1,y1,x2,y2
Settings follow each codebase's own validation (test.py / val.py): letterbox to --img, fp16, conf 0.001, IoU 0.6,
multi-label NMS (val.py also caps at 300 detections per image), DIoU-NMS when --diou-nms (P2 improved model).
Images are letterboxed one at a time to a square --img (no rectangular batching), so mAP can differ slightly from
the stock val script; dump_check in cross_eval.py measures that on each home dataset.

usage (from the codebase dir, i.e. code/gbh-yolov5 or code/yolov5-v7.0):
  <env python> ../repro/cross/dump_v5.py --weights W --list L --img 960 --out preds.csv [--diou-nms]
"""
import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from models.experimental import attempt_load  # noqa: E402

try:  # yolov5 v7.0
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_boxes
    V7 = True
except ImportError:  # GBH fork (yolov5 v5.0 era)
    from utils.datasets import letterbox
    from utils.general import non_max_suppression, scale_coords as scale_boxes
    V7 = False

ap = argparse.ArgumentParser()
ap.add_argument('--weights', required=True)
ap.add_argument('--list', required=True, help='text file, one absolute image path per line')
ap.add_argument('--img', type=int, required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--conf', type=float, default=0.001)
ap.add_argument('--iou', type=float, default=0.6)
ap.add_argument('--diou-nms', action='store_true')
a = ap.parse_args()

dev = torch.device('cuda:0')
model = attempt_load(a.weights, device=dev) if V7 else attempt_load(a.weights, map_location=dev)
model.half()
stride = int(model.stride.max())
names = model.module.names if hasattr(model, 'module') else model.names
names = list(names.values()) if isinstance(names, dict) else list(names)
paths = [l.strip() for l in open(a.list) if l.strip()]

model(torch.zeros(1, 3, a.img, a.img, device=dev).half())  # warm-up
t_inf = t_nms = 0.0
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with open(a.out, 'w', newline='') as f, torch.no_grad():
    w = csv.writer(f)
    w.writerow(['image', 'cls', 'conf', 'x1', 'y1', 'x2', 'y2'])
    for p in paths:
        im0 = cv2.imread(p)  # BGR, grayscale images replicated to 3 channels (as the codebases' loaders do)
        im = letterbox(im0, (a.img, a.img), auto=False, stride=stride)[0]
        im = np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1))
        x = torch.from_numpy(im).to(dev).half()[None] / 255.0
        torch.cuda.synchronize(); t0 = time.time()
        pred = model(x)[0]
        torch.cuda.synchronize(); t1 = time.time()
        kw = dict(conf_thres=a.conf, iou_thres=a.iou, multi_label=True)
        if V7:
            kw.update(max_det=300, diou=a.diou_nms)
        elif a.diou_nms:
            sys.exit('DIoU-NMS exists only in the patched v7.0 codebase')
        det = non_max_suppression(pred, **kw)[0]
        torch.cuda.synchronize(); t_nms += time.time() - t1; t_inf += t1 - t0
        if len(det):
            det[:, :4] = scale_boxes(x.shape[2:], det[:, :4], im0.shape).round()
            stem = os.path.splitext(os.path.basename(p))[0]
            for *xyxy, conf, c in det.tolist():
                w.writerow([stem, int(c), f'{conf:.5f}'] + [f'{v:.1f}' for v in xyxy])

meta = {'weights': a.weights, 'list': a.list, 'n_images': len(paths), 'img': a.img, 'conf': a.conf, 'iou': a.iou,
        'diou_nms': a.diou_nms, 'names': names, 'ms_inference': 1000 * t_inf / len(paths),
        'ms_nms': 1000 * t_nms / len(paths), 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__}
json.dump(meta, open(os.path.splitext(a.out)[0] + '.json', 'w'), indent=1)
print(json.dumps(meta))
