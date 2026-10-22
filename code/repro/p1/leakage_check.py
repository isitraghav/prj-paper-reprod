"""Train/val leakage check for PV-Multi-Defect (P1): duplicate images inside the dataset, and how much of our val split
has a (near-)copy in train. Re-scores the P1 models' val predictions (results/cross/preds, inference only) on the
val images with and without such a copy.

Near-duplicate = average hash (16x16 grayscale > mean, 256 bits) within --max-dist bits of some train image.
Byte-identical copies are distance 0. Labels of duplicate pairs are compared too.

usage (repo root): envs/p1/Scripts/python.exe -I code/repro/p1/leakage_check.py [--max-dist 15]
Writes results/p1/leakage.json and prints a summary.
"""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/repro/cross'))
from common import load_gt, load_preds, stem  # noqa: E402
from detmetrics import det_metrics, det_stats  # noqa: E402

SPLITS = ROOT / 'datasets/splits/pv_multi_defect'
IMAGES = ROOT / 'datasets/raw/PV-Multi-Defect/JPEGImages'
MODELS = ['p1_yolov5s', 'p1_gbh']

ap = argparse.ArgumentParser()
ap.add_argument('--max-dist', type=int, default=15)
ap.add_argument('--boot', type=int, default=1000)
a = ap.parse_args()

split = {s: [stem(l.strip()) for l in open(SPLITS / f'{s}.txt') if l.strip()] for s in ('train', 'val')}
side = {k: s for s, ks in split.items() for k in ks}


def ahash(k):
    g = np.asarray(Image.open(IMAGES / f'{k}.jpg').convert('L').resize((16, 16), Image.BILINEAR), np.float32)
    return (g > g.mean()).ravel()


md5 = defaultdict(list)
for k in side:
    md5[hashlib.md5((IMAGES / f'{k}.jpg').read_bytes()).hexdigest()].append(k)
groups = [g for g in md5.values() if len(g) > 1]

gt = load_gt('pv')  # val labels in pixels; train labels read the same way below
lab_dir = ROOT / 'datasets/pv_multi_defect_yolo/labels'


def yolo_labels(k):
    for f in (lab_dir / 'train' / f'{k}.txt', lab_dir / 'val' / f'{k}.txt', lab_dir / f'{k}.txt'):
        if f.exists():
            return sorted(tuple(round(float(x), 4) for x in l.split()) for l in open(f) if l.strip())
    return None


same_labels = sum(len({str(yolo_labels(k)) for k in g}) == 1 for g in groups)

H = {k: ahash(k) for k in side}
tr = np.array([H[k] for k in split['train']])
dist = {k: int((tr != H[k]).sum(1).min()) for k in split['val']}
leaky = [k for k in split['val'] if dist[k] <= a.max_dist]
clean = [k for k in split['val'] if dist[k] > a.max_dist]

out = {'images': len(side), 'duplicate_groups': len(groups), 'images_in_duplicate_groups': sum(map(len, groups)),
       'cross_split_groups': sum(len({side[k] for k in g}) > 1 for g in groups),
       'groups_with_identical_labels': same_labels, 'max_dist': a.max_dist,
       'val': len(split['val']), 'val_exact_copy_in_train': sum(d == 0 for d in dist.values()),
       'val_near_copy_in_train': len(leaky), 'models': {}}

rng = np.random.default_rng(0)
for m in MODELS:
    per = det_stats(load_preds(m, 'pv'), gt, split['val'])
    r = {name: det_metrics(per, keys)['mAP50'] for name, keys in (('all', split['val']), ('clean', clean), ('leaky', leaky))}
    boot = [det_metrics(per, list(rng.choice(clean, len(clean))))['mAP50'] for _ in range(a.boot)]
    r['clean_ci'] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    out['models'][m] = {k: (float(v) if not isinstance(v, list) else v) for k, v in r.items()}

(ROOT / 'results/p1').mkdir(parents=True, exist_ok=True)
json.dump(out, open(ROOT / 'results/p1/leakage.json', 'w'), indent=1)
print(f"{out['images']} images: {out['duplicate_groups']} byte-identical groups ({out['images_in_duplicate_groups']} images), "
      f"{out['cross_split_groups']} straddle train/val, {same_labels} with identical labels")
print(f"val {out['val']}: exact copy in train {out['val_exact_copy_in_train']}, near copy (<= {a.max_dist} bits) {len(leaky)}")
for m, r in out['models'].items():
    print(f"{m:12s} mAP@.5 (our evaluator) all {r['all']:.3f} | clean {r['clean']:.3f} {r['clean_ci']} (n {len(clean)}) | "
          f"leaky {r['leaky']:.3f} (n {len(leaky)})")
