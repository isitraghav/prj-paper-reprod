"""Shared definitions for the cross-dataset comparison (P1 GBH-YOLOv5, P2 Improved-YOLOv5/YOLOv8, P3 VGG-19 CNN).

Datasets (each paper's own evaluation split):
  pv    PV-Multi-Defect val, 222 images 600x600 (P1).  Defect classes: all 5.
  eldds ELDDS1400C5 test, 190 images 640x640 (P2).     Defect classes: 1-5; class 0 'Examined' is the module outline,
                                                       not a defect -> dropped from GT and from P2 predictions.
  elpv  ELPV test, 656 cells 300x300 (P3).             Label: defective iff p > 0 (no boxes).

Tiles (the shared defective-vs-functional task, so the cell classifier can run on module images):
  pv    fixed 4x4 grid over the image (150 px tiles).
  eldds 6x6 grid over the module region = the image's GT 'Examined' box (union if several; whole image if none).
  elpv  every cell is one tile.
  A tile is defective iff the centre of a GT defect box lies inside it. A detector's tile score is the max confidence
  of its defect detections whose centre lies inside it (0 if none); the CNN's tile score is its output on the tile
  crop resized to 300x300.
"""
import csv
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results' / 'cross'
ELPV_DATA = ROOT / 'datasets/raw/elpv-dataset/src/elpv_dataset/data'

DATASETS = ['pv', 'eldds', 'elpv']
DS_LABEL = {'pv': 'PV-Multi-Defect (P1)', 'eldds': 'ELDDS1400C5 (P2)', 'elpv': 'ELPV (P3)'}
GRID = {'pv': 4, 'eldds': 6}

# model id -> (label, home dataset, paper, defect classes it predicts (None = all))
MODELS = {
    'p1_yolov5s':  ('P1 YOLOv5s', 'pv', 'P1', None),
    'p1_gbh':      ('P1 GBH-YOLOv5', 'pv', 'P1', None),
    'p2_v5s':      ('P2 YOLOv5s', 'eldds', 'P2', [1, 2, 3, 4, 5]),
    'p2_improved': ('P2 YOLOv5s+GAM+ASFF+DIoU', 'eldds', 'P2', [1, 2, 3, 4, 5]),
    'p2_v8s':      ('P2 YOLOv8s', 'eldds', 'P2', [1, 2, 3, 4, 5]),
    'p3_cnn':      ('P3 VGG-19 CNN', 'elpv', 'P3', None),
}
DETECTORS = [m for m in MODELS if m != 'p3_cnn']
# categorical slots in fixed order (dataviz reference palette, light mode)
COLOR = dict(zip(MODELS, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']))


def image_list(ds):
    return [l.strip() for l in open(OUT / 'lists' / f'{ds}.txt') if l.strip()]


def stem(p):
    return os.path.splitext(os.path.basename(p))[0]


def label_path(img):  # YOLO convention, as both codebases resolve it
    return img.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'


def image_size(ds):
    return {'pv': (600, 600), 'eldds': (640, 640), 'elpv': (300, 300)}[ds]  # (w, h); checked in make_tiles


def load_gt(ds):
    """-> {stem: (n,5) array [cls, x1, y1, x2, y2] in pixels}, all classes as labelled."""
    W, H = image_size(ds)
    gt = {}
    for p in image_list(ds):
        f = label_path(p)
        rows = [l.split() for l in open(f) if l.strip()] if os.path.exists(f) else []
        a = np.array(rows, float).reshape(-1, 5)
        if len(a):
            c, x, y, w, h = a.T
            a = np.stack([c, (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H], 1)
        gt[stem(p)] = a
    return gt


def defect_gt(ds, gt):
    """Drop non-defect GT classes (ELDDS 'Examined')."""
    return {k: (v[v[:, 0] != 0] if ds == 'eldds' else v) for k, v in gt.items()}


def load_preds(model, ds):
    """-> {stem: (n,6) array [cls, conf, x1, y1, x2, y2]} for every image of the list (empty if no detections)."""
    out = {stem(p): [] for p in image_list(ds)}
    with open(OUT / 'preds' / f'{model}__{ds}.csv') as f:
        for r in csv.DictReader(f):
            out[r['image']].append([float(r[k]) for k in ('cls', 'conf', 'x1', 'y1', 'x2', 'y2')])
    return {k: np.array(v, float).reshape(-1, 6) for k, v in out.items()}


def defect_preds(model, preds):
    keep = MODELS[model][3]
    if keep is None:
        return preds
    return {k: v[np.isin(v[:, 0], keep)] for k, v in preds.items()}


def elpv_labels():
    """-> {stem: (p, type)} for the ELPV test cells."""
    lab = {}
    for l in (ELPV_DATA / 'labels.csv').read_text().splitlines():
        if l.strip():
            path, p, t = l.split()
            lab[stem(path)] = (float(p), t)
    return {stem(p): lab[stem(p)] for p in image_list('elpv')}


def make_tiles(grid=None):
    """grid: {ds: n} override of GRID (sensitivity runs). -> list of dicts: ds, image, path, r, c, x1, y1, x2, y2, label (1 defective / 0 functional), p (ELPV), type."""
    tiles = []
    for ds in ('pv', 'eldds'):
        gt = load_gt(ds)
        n = (grid or GRID)[ds]
        for p in image_list(ds):
            s = stem(p)
            g = gt[s]
            W, H = image_size(ds)
            ex = g[g[:, 0] == 0] if ds == 'eldds' else np.zeros((0, 5))
            if len(ex):
                X1, Y1, X2, Y2 = ex[:, 1].min(), ex[:, 2].min(), ex[:, 3].max(), ex[:, 4].max()
            else:
                X1, Y1, X2, Y2 = 0, 0, W, H
            X1, Y1, X2, Y2 = max(0, X1), max(0, Y1), min(W, X2), min(H, Y2)
            d = defect_gt(ds, {s: g})[s]
            cx, cy = (d[:, 1] + d[:, 3]) / 2, (d[:, 2] + d[:, 4]) / 2
            xs, ys = np.linspace(X1, X2, n + 1), np.linspace(Y1, Y2, n + 1)
            for r in range(n):
                for c in range(n):
                    x1, x2, y1, y2 = xs[c], xs[c + 1], ys[r], ys[r + 1]
                    inside = (cx >= x1) & (cx < x2) & (cy >= y1) & (cy < y2)
                    tiles.append(dict(ds=ds, image=s, path=p, r=r, c=c, x1=x1, y1=y1, x2=x2, y2=y2,
                                      label=int(inside.any()), p='', type=''))
    lab = elpv_labels()
    for p in image_list('elpv'):
        s = stem(p)
        pv, t = lab[s]
        tiles.append(dict(ds='elpv', image=s, path=p, r=0, c=0, x1=0, y1=0, x2=300, y2=300,
                          label=int(pv > 0), p=pv, type=t))
    return tiles


def confidence_weight(p):  # P3 Table 2: p=1/3 -> 0.33, 2/3 -> 0.67, else 1
    p = np.asarray(p, float)
    return np.where(np.isclose(p, 1 / 3, atol=1e-3), 0.33, np.where(np.isclose(p, 2 / 3, atol=1e-3), 0.67, 1.0))
