"""KAZE keypoints per ELPV training cell with OpenCV's default settings (the paper: "OpenCV defaults").
Compares OpenCV builds: run once in env p3svm (3.4.18) and once in the 3.3.1 container (docker/Dockerfile).

usage (repo root): <python> code/repro/p3/kaze_count.py   -> prints one JSON line
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'datasets/raw/elpv-dataset/src/elpv_dataset/data'
paths = (ROOT / 'datasets/splits/elpv/train.txt').read_text().split()
n = np.array([len(cv2.KAZE_create().detect(cv2.imread(str(DATA / p), cv2.IMREAD_GRAYSCALE), None)) for p in paths])
sys.stdout.write(json.dumps({'opencv': cv2.__version__, 'cells': len(n), 'mean_keypoints': float(n.mean()),
                             'median': float(np.median(n)), 'zero_keypoint_cells': int((n == 0).sum())}) + '\n')
