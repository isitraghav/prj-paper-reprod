"""Score every model on every dataset with one evaluator (inference outputs from run_dumps.sh and cnn_tiles.py).

1. Detection (detectors x {pv, eldds}): mAP@.5, mAP@.5:.95, P, R. Re-implementation of yolov5 v7.0 utils/metrics.py
   (ap_per_class: 101-point interpolated AP, P/R at the max of the 0.1-smoothed mean F1; val.py process_batch matching),
   so every model is scored by the same code.
   - 'home': class-aware, each model on its own dataset with its own classes = the paper metric (sanity check against
     our official val logs).
   - 'agnostic': all defect classes merged into one 'defect' class (ELDDS 'Examined' dropped from GT and P2 predictions),
     class-agnostic NMS at the model's own NMS IoU, max 300 detections. Comparable across all datasets and models.
   - per-GT-class recall at IoU 0.5 for detections with conf >= 0.25 (which kinds of defect a foreign model finds).
2. Defective vs functional tiles (all 6 models x {pv, eldds, elpv}; tiles from common.py): AUC, accuracy, macro-F1,
   defective-class precision / recall / F1. Fixed, pre-declared thresholds: CNN 0.5 (P3), detectors conf 0.25
   (yolov5 detect.py / ultralytics predict default). ELPV additionally with P3's confidence weights.
3. 95 % CIs: 1000 bootstrap resamples over images (tiles of one image stay together), seed 0.

usage (repo root): envs/p3cnn/Scripts/python.exe -I code/repro/cross/cross_eval.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve  # noqa: E402

from detmetrics import box_iou, ci, det_metrics, det_stats, nms_agnostic  # noqa: E402
from common import (DATASETS, DETECTORS, MODELS, OUT, confidence_weight, defect_gt, defect_preds, load_gt,  # noqa: E402
                    load_preds)

NMS_IOU = {m: (0.7 if m == 'p2_v8s' else 0.6) for m in DETECTORS}
DET_THR, CNN_THR, NBOOT = 0.25, 0.5, 1000
rng = np.random.default_rng(0)


# ------------------------------------------------------------------ 1. detection
det_rows, recall_rows, pr_curves = [], [], {}
GT = {ds: load_gt(ds) for ds in ('pv', 'eldds')}
for m in DETECTORS:
    home = MODELS[m][1]
    for ds in ('pv', 'eldds'):
        preds = load_preds(m, ds)
        keys = sorted(preds)
        if ds == home:  # class-aware, paper metric
            r = det_metrics(det_stats(preds, GT[ds], keys), keys)
            det_rows.append(dict(model=m, ds=ds, mode='home', **{k: r[k] for k in ('P', 'R', 'mAP50', 'mAP')},
                                 ap50_per_class=r['ap50_per_class']))
        dp = {k: nms_agnostic(np.c_[np.zeros(len(v)), v[:, 1:]], NMS_IOU[m]) for k, v in defect_preds(m, preds).items()}
        dg_full = defect_gt(ds, GT[ds])
        dg = {k: np.c_[np.zeros(len(v)), v[:, 1:]] for k, v in dg_full.items()}
        per = det_stats(dp, dg, keys)
        r = det_metrics(per, keys, curve=True)
        boot = [det_metrics(per, list(rng.choice(keys, len(keys)))) ['mAP50'] for _ in range(NBOOT)]
        det_rows.append(dict(model=m, ds=ds, mode='agnostic', **{k: r[k] for k in ('P', 'R', 'mAP50', 'mAP')},
                             mAP50_ci=ci(boot)))
        pr_curves[f'{m}__{ds}'] = [r['pr_curve'][0][::max(1, len(r['pr_curve'][0]) // 400)].tolist(),
                                   r['pr_curve'][1][::max(1, len(r['pr_curve'][1]) // 400)].tolist()]
        # recall per GT class at IoU .5, conf >= DET_THR (class-agnostic matching)
        found = {}
        for k in keys:
            g, d = dg_full[k], dp[k][dp[k][:, 1] >= DET_THR]
            hit = (box_iou(g[:, 1:], d[:, 2:]) >= 0.5).any(1) if len(g) and len(d) else np.zeros(len(g), bool)
            for c, h in zip(g[:, 0].astype(int), hit):
                found.setdefault(c, []).append(h)
        for c, h in sorted(found.items()):
            recall_rows.append(dict(model=m, ds=ds, gt_cls=int(c), n=len(h), recall=float(np.mean(h))))
        print(f'{m:12s} {ds:6s} agnostic mAP50 {r["mAP50"]:.3f} [{ci(boot)[0]:.3f}, {ci(boot)[1]:.3f}] '
              f'mAP {r["mAP"]:.3f} P {r["P"]:.3f} R {r["R"]:.3f}', flush=True)
for row in det_rows:
    if row['mode'] == 'home':
        print(f'{row["model"]:12s} {row["ds"]:6s} HOME class-aware mAP50 {row["mAP50"]:.3f} mAP {row["mAP"]:.3f} '
              f'P {row["P"]:.3f} R {row["R"]:.3f}')

# ------------------------------------------------------------------ 2. tiles
tiles = list(csv.DictReader(open(OUT / 'tiles.csv')))
cnn = {(t['ds'], t['image'], t['r'], t['c']): float(t['score']) for t in csv.DictReader(open(OUT / 'preds/p3_cnn__tiles.csv'))}
scores = {'p3_cnn': np.array([cnn[(t['ds'], t['image'], t['r'], t['c'])] for t in tiles])}
for m in DETECTORS:
    s = np.zeros(len(tiles))
    for ds in DATASETS:
        dp = defect_preds(m, load_preds(m, ds))
        for i, t in enumerate(tiles):
            if t['ds'] != ds:
                continue
            d = dp[t['image']]
            if not len(d):
                continue
            cx, cy = (d[:, 2] + d[:, 4]) / 2, (d[:, 3] + d[:, 5]) / 2
            if ds == 'elpv':
                inside = np.ones(len(d), bool)  # the cell is the tile
            else:
                inside = (cx >= float(t['x1'])) & (cx < float(t['x2'])) & (cy >= float(t['y1'])) & (cy < float(t['y2']))
            if inside.any():
                s[i] = d[inside, 1].max()
    scores[m] = s

tile_rows, roc_curves = [], {}
ds_col = np.array([t['ds'] for t in tiles])
img_col = np.array([t['image'] for t in tiles])
y_all = np.array([int(t['label']) for t in tiles])


def cls_metrics(y, s, thr, w=None):
    yhat = (s >= thr).astype(int)
    kw = {} if w is None else {'sample_weight': w}
    return dict(auc=roc_auc_score(y, s, **kw), acc=accuracy_score(y, yhat, **kw),
                f1_macro=f1_score(y, yhat, average='macro', **kw),
                precision=precision_score(y, yhat, zero_division=0, **kw), recall=recall_score(y, yhat, **kw),
                f1_defective=f1_score(y, yhat, **kw), pos_rate=float(yhat.mean()))


for m in MODELS:
    thr = CNN_THR if m == 'p3_cnn' else DET_THR
    for ds in DATASETS:
        sel = ds_col == ds
        y, s, im = y_all[sel], scores[m][sel], img_col[sel]
        r = cls_metrics(y, s, thr)
        groups = {k: np.nonzero(im == k)[0] for k in np.unique(im)}
        gk = list(groups)
        bauc, bf1 = [], []
        for _ in range(NBOOT):
            idx = np.concatenate([groups[k] for k in rng.choice(gk, len(gk))])
            if len(np.unique(y[idx])) < 2:
                continue
            bauc.append(roc_auc_score(y[idx], s[idx]))
            bf1.append(f1_score(y[idx], (s[idx] >= thr).astype(int), average='macro'))
        row = dict(model=m, ds=ds, n=int(sel.sum()), n_defective=int(y.sum()), threshold=thr, **r,
                   auc_ci=ci(bauc), f1_macro_ci=ci(bf1))
        if ds == 'elpv':
            p = np.array([float(t['p']) for t in np.array(tiles, dtype=object)[sel]])
            rw = cls_metrics(y, s, thr, w=confidence_weight(p))
            row.update({f'{k}_w': v for k, v in rw.items() if k in ('auc', 'acc', 'f1_macro')})
        tile_rows.append(row)
        fpr, tpr, _ = roc_curve(y, s)
        step = max(1, len(fpr) // 400)
        roc_curves[f'{m}__{ds}'] = [fpr[::step].tolist() + [1.0], tpr[::step].tolist() + [1.0]]
        print(f'{m:12s} {ds:6s} tiles AUC {r["auc"]:.3f} [{ci(bauc)[0]:.3f}, {ci(bauc)[1]:.3f}] acc {r["acc"]:.3f} '
              f'F1m {r["f1_macro"]:.3f} P {r["precision"]:.3f} R {r["recall"]:.3f}'
              + (f'  acc_w {row["acc_w"]:.4f} F1_w {row["f1_macro_w"]:.4f} AUC_w {row["auc_w"]:.4f}' if ds == 'elpv' else ''),
              flush=True)

speed = {}
for m in DETECTORS:
    for ds in DATASETS:
        j = json.load(open(OUT / 'preds' / f'{m}__{ds}.json'))
        speed[f'{m}__{ds}'] = dict(ms_inference=j['ms_inference'], ms_nms=j['ms_nms'], img=j['img'])
speed['p3_cnn__tiles'] = json.load(open(OUT / 'preds/p3_cnn__tiles.json'))
json.dump(dict(detection=det_rows, recall_by_gt_class=recall_rows, tiles=tile_rows, speed=speed,
               settings=dict(det_threshold=DET_THR, cnn_threshold=CNN_THR, nms_iou=NMS_IOU, n_boot=NBOOT)),
          open(OUT / 'metrics.json', 'w'), indent=1)
json.dump(dict(pr=pr_curves, roc=roc_curves), open(OUT / 'curves.json', 'w'))
print('wrote', OUT / 'metrics.json')
