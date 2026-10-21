"""Cross-dataset inference for P2's YOLOv8s (ultralytics 8.0.99), same output format as dump_v5.py.

Settings follow ultralytics' own validation: imgsz 640, fp16, conf 0.001, IoU 0.7, max 300 detections, multi-label NMS.

usage: <p2 python> code/repro/cross/dump_v8.py --weights W --list L --img 640 --out preds.csv
"""
import argparse
import csv
import json
import os
import time

import cv2
import numpy as np
import torch
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.data.augment import LetterBox
from ultralytics.yolo.utils import ops

ap = argparse.ArgumentParser()
ap.add_argument('--weights', required=True)
ap.add_argument('--list', required=True)
ap.add_argument('--img', type=int, default=640)
ap.add_argument('--out', required=True)
ap.add_argument('--conf', type=float, default=0.001)
ap.add_argument('--iou', type=float, default=0.7)
a = ap.parse_args()

dev = torch.device('cuda:0')
model = AutoBackend(a.weights, device=dev, fp16=True)
model.eval()
lb = LetterBox((a.img, a.img), auto=False, stride=int(model.stride))
paths = [l.strip() for l in open(a.list) if l.strip()]
model(torch.zeros(1, 3, a.img, a.img, device=dev).half())
t_inf = t_nms = 0.0
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with open(a.out, 'w', newline='') as f, torch.no_grad():
    w = csv.writer(f)
    w.writerow(['image', 'cls', 'conf', 'x1', 'y1', 'x2', 'y2'])
    for p in paths:
        im0 = cv2.imread(p)
        im = lb(image=im0)
        im = np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1))
        x = torch.from_numpy(im).to(dev).half()[None] / 255.0
        torch.cuda.synchronize(); t0 = time.time()
        pred = model(x)
        torch.cuda.synchronize(); t1 = time.time()
        det = ops.non_max_suppression(pred, a.conf, a.iou, multi_label=True, max_det=300)[0]
        torch.cuda.synchronize(); t_nms += time.time() - t1; t_inf += t1 - t0
        if len(det):
            det[:, :4] = ops.scale_boxes(x.shape[2:], det[:, :4], im0.shape).round()
            stem = os.path.splitext(os.path.basename(p))[0]
            for *xyxy, conf, c in det.tolist():
                w.writerow([stem, int(c), f'{conf:.5f}'] + [f'{v:.1f}' for v in xyxy])

meta = {'weights': a.weights, 'list': a.list, 'n_images': len(paths), 'img': a.img, 'conf': a.conf, 'iou': a.iou,
        'diou_nms': False, 'names': list(model.names.values()), 'ms_inference': 1000 * t_inf / len(paths),
        'ms_nms': 1000 * t_nms / len(paths), 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__}
json.dump(meta, open(os.path.splitext(a.out)[0] + '.json', 'w'), indent=1)
print(json.dumps(meta))
