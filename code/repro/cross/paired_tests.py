"""Paired bootstrap tests for the architecture comparisons in COMPARISON.md, plus per-class AP CIs for the
class-mapped detection (class_map.py).

Two models scored on the same images are compared on the same bootstrap resamples (1000 over images, seed 0; the
tiles of an image stay together), so the CI is on the *difference* (overlapping per-model CIs are not a test).
Pairs: GBH-YOLOv5 - P1 YOLOv5s; P2 Improved - P2 YOLOv5s; P2 YOLOv8s - P2 YOLOv5s; P2 YOLOv8s - P2 Improved.
Metrics: home class-aware mAP@.5 (paper classes; for P2 also defect classes only), class-agnostic mAP@.5 (home and
cross), class-mapped mAP@.5 (home and cross), tile AUC on all three datasets. Same scoring code and settings as
cross_eval.py / class_map.py.

usage (repo root): envs/p3cnn/Scripts/python.exe -I code/repro/cross/paired_tests.py
Writes results/cross/paired_tests.json, tables_paired.md.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from common import (DATASETS, DETECTORS, DS_LABEL, MODELS, OUT, SHARED, defect_gt, defect_preds, load_gt,  # noqa: E402
                    load_preds, mapped_preds, relabel)
from detmetrics import ci, det_metrics, det_stats, nms_agnostic  # noqa: E402

NMS_IOU = {m: (0.7 if m == 'p2_v8s' else 0.6) for m in DETECTORS}
NBOOT = 1000
PAIRS = [('p1_gbh', 'p1_yolov5s'), ('p2_improved', 'p2_v5s'), ('p2_v8s', 'p2_v5s'), ('p2_v8s', 'p2_improved')]
GT = {ds: load_gt(ds) for ds in ('pv', 'eldds')}
DG = {ds: defect_gt(ds, GT[ds]) for ds in GT}
AGN = lambda a: np.c_[np.zeros(len(a)), a[:, 1:]]  # noqa: E731


def det_inputs(m, ds, kind):
    """-> per-image stats for det_metrics under one scoring mode."""
    preds = load_preds(m, ds)
    keys = sorted(preds)
    if kind == 'home_paper':
        return det_stats(preds, GT[ds], keys)
    if kind == 'home_defect':
        return det_stats(defect_preds(m, preds), DG[ds], keys)
    if kind == 'agnostic':
        dp = {k: nms_agnostic(AGN(v), NMS_IOU[m]) for k, v in defect_preds(m, preds).items()}
        return det_stats(dp, {k: AGN(v) for k, v in DG[ds].items()}, keys)
    if kind == 'mapped':
        return det_stats(mapped_preds(m, preds, 'fine', NMS_IOU[m]),
                         {k: relabel(v, ds, 'fine') for k, v in DG[ds].items()}, keys)


def tile_scores(m, tiles):
    s = np.zeros(len(tiles))
    for ds in DATASETS:
        dp = defect_preds(m, load_preds(m, ds))
        for i, t in enumerate(tiles):
            if t['ds'] != ds or not len(dp[t['image']]):
                continue
            d = dp[t['image']]
            cx, cy = (d[:, 2] + d[:, 4]) / 2, (d[:, 3] + d[:, 5]) / 2
            ins = np.ones(len(d), bool) if ds == 'elpv' else \
                (cx >= float(t['x1'])) & (cx < float(t['x2'])) & (cy >= float(t['y1'])) & (cy < float(t['y2']))
            if ins.any():
                s[i] = d[ins, 1].max()
    return s


def summary(point, boot):
    b = np.asarray(boot)
    p = min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))  # two-sided bootstrap p
    return dict(diff=float(point), ci=ci(b), p=float(p))


rng = np.random.default_rng(0)
tests = []
for a, b in PAIRS:
    home = MODELS[a][1]
    cross = 'eldds' if home == 'pv' else 'pv'
    cells = [('home_paper', home), ('agnostic', home), ('agnostic', cross), ('mapped', home), ('mapped', cross)]
    if home == 'eldds':
        cells.insert(1, ('home_defect', home))
    for kind, ds in cells:
        pa, pb = det_inputs(a, ds, kind), det_inputs(b, ds, kind)
        keys = sorted(pa)
        point = det_metrics(pa, keys)['mAP50'] - det_metrics(pb, keys)['mAP50']
        boot = []
        for _ in range(NBOOT):
            kk = list(rng.choice(keys, len(keys)))
            boot.append(det_metrics(pa, kk)['mAP50'] - det_metrics(pb, kk)['mAP50'])
        tests.append(dict(a=a, b=b, metric=f'{kind} mAP@.5', ds=ds, **summary(point, boot)))
        r = tests[-1]
        print(f'{a:12s} - {b:12s} {kind:12s} {ds:6s} {r["diff"]:+.3f} [{r["ci"][0]:+.3f}, {r["ci"][1]:+.3f}] '
              f'p {r["p"]:.3f}', flush=True)

tiles = list(csv.DictReader(open(OUT / 'tiles.csv')))
y_all = np.array([int(t['label']) for t in tiles])
ds_col, img_col = np.array([t['ds'] for t in tiles]), np.array([t['image'] for t in tiles])
TS = {m: tile_scores(m, tiles) for m in {x for p in PAIRS for x in p}}
for a, b in PAIRS:
    for ds in DATASETS:
        sel = np.nonzero(ds_col == ds)[0]
        y, sa, sb, im = y_all[sel], TS[a][sel], TS[b][sel], img_col[sel]
        groups = {k: np.nonzero(im == k)[0] for k in np.unique(im)}
        gk = list(groups)
        point = roc_auc_score(y, sa) - roc_auc_score(y, sb)
        boot = []
        for _ in range(NBOOT):
            idx = np.concatenate([groups[k] for k in rng.choice(gk, len(gk))])
            if len(np.unique(y[idx])) == 2:
                boot.append(roc_auc_score(y[idx], sa[idx]) - roc_auc_score(y[idx], sb[idx]))
        tests.append(dict(a=a, b=b, metric='tile AUC', ds=ds, **summary(point, boot)))
        r = tests[-1]
        print(f'{a:12s} - {b:12s} tile AUC     {ds:6s} {r["diff"]:+.3f} [{r["ci"][0]:+.3f}, {r["ci"][1]:+.3f}] '
              f'p {r["p"]:.3f}', flush=True)

# per-class AP@.5 CIs for the class-mapped detection
class_ci = []
for m in DETECTORS:
    for ds in ('pv', 'eldds'):
        per = det_inputs(m, ds, 'mapped')
        keys = sorted(per)
        ap = det_metrics(per, keys)['ap50_per_class']
        boot = {c: [] for c in ap}
        for _ in range(NBOOT):
            bb = det_metrics(per, list(rng.choice(keys, len(keys))))['ap50_per_class']
            for c in ap:
                boot[c].append(bb.get(c, 0.0))  # class absent from a resample: AP 0 (rare: PV crack has 18 boxes)
        for c in ap:
            class_ci.append(dict(model=m, ds=ds, home=ds == MODELS[m][1], cls=SHARED['fine'][c], ap50=ap[c],
                                 ci=ci(boot[c])))
            print(f'{m:12s} {ds:6s} {SHARED["fine"][c]:6s} AP50 {ap[c]:.3f} [{ci(boot[c])[0]:.3f}, {ci(boot[c])[1]:.3f}]',
                  flush=True)

json.dump(dict(paired=tests, mapped_class_ap_ci=class_ci, settings=dict(n_boot=NBOOT, seed=0, nms_iou=NMS_IOU)),
          open(OUT / 'paired_tests.json', 'w'), indent=1)

L = ['### Paired bootstrap: difference A - B (same 1000 image resamples), 95 % CI, two-sided bootstrap p\n',
     '| A - B | metric | dataset | difference [95 % CI] | p |', '|---|---|---|---|---|']
for r in tests:
    sig = ' **' if r['ci'][0] > 0 or r['ci'][1] < 0 else ''
    L.append(f'| {MODELS[r["a"]][0]} - {MODELS[r["b"]][0]} | {r["metric"]} | {DS_LABEL[r["ds"]]} | '
             f'{sig.strip()}{r["diff"]:+.3f}{sig.strip()} [{r["ci"][0]:+.3f}, {r["ci"][1]:+.3f}] | {r["p"]:.3f} |')
L += ['\nBold = CI excludes 0.\n', '### Class-mapped AP@.5 per shared class, 95 % bootstrap CI\n',
      '| Model | Dataset | | crack | line | dark |', '|---|---|---|---|---|---|']
by = {(r['model'], r['ds'], r['cls']): r for r in class_ci}
for m in DETECTORS:
    for ds in (MODELS[m][1], 'eldds' if MODELS[m][1] == 'pv' else 'pv'):
        cells = [by[(m, ds, c)] for c in SHARED['fine']]
        L.append(f'| {MODELS[m][0]} | {DS_LABEL[ds]} | {"home" if ds == MODELS[m][1] else "cross"} | '
                 + ' | '.join(f'{r["ap50"]:.3f} [{r["ci"][0]:.3f}, {r["ci"][1]:.3f}]' for r in cells) + ' |')
(OUT / 'tables_paired.md').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
