"""Sensitivity checks for the two free choices of the cross-dataset protocol (main protocol: cross_eval.py).

S1 input size: main runs use each model's own paper size (P1 960, P2 640). Here the detectors also run on the foreign
   dataset at that dataset's paper size (P1 models on ELDDS @640, P2 models on PV @960; preds_sens/).
S2 tile size: ELDDS 12x12 tiles over the module (about one or two cells per tile, closer to an ELPV cell) instead of 6x6.
Same metrics, thresholds and bootstrap as cross_eval.py.

usage (repo root): envs/p3cnn/Scripts/python.exe -I code/repro/cross/sensitivity.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402

from common import DETECTORS, MODELS, OUT, defect_gt, defect_preds, load_gt, load_preds  # noqa: E402
from detmetrics import ci, det_metrics, det_stats, nms_agnostic  # noqa: E402

NMS_IOU = {m: (0.7 if m == 'p2_v8s' else 0.6) for m in DETECTORS}
DET_THR, CNN_THR, NBOOT = 0.25, 0.5, 1000
rng = np.random.default_rng(0)
SENS = OUT / 'preds_sens'


def preds_from(path, ds):
    from common import image_list, stem
    out = {stem(p): [] for p in image_list(ds)}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r['image']].append([float(r[k]) for k in ('cls', 'conf', 'x1', 'y1', 'x2', 'y2')])
    return {k: np.array(v, float).reshape(-1, 6) for k, v in out.items()}


def agnostic(m, ds, preds):
    keys = sorted(preds)
    dp = {k: nms_agnostic(np.c_[np.zeros(len(v)), v[:, 1:]], NMS_IOU[m]) for k, v in defect_preds(m, preds).items()}
    dg = {k: np.c_[np.zeros(len(v)), v[:, 1:]] for k, v in defect_gt(ds, load_gt(ds)).items()}
    per = det_stats(dp, dg, keys)
    r = det_metrics(per, keys)
    boot = [det_metrics(per, list(rng.choice(keys, len(keys))))['mAP50'] for _ in range(NBOOT)]
    return dict(mAP50=r['mAP50'], mAP=r['mAP'], P=r['P'], R=r['R'], mAP50_ci=ci(boot))


def tile_scores(tiles, preds):
    s = np.zeros(len(tiles))
    for i, t in enumerate(tiles):
        d = preds[t['image']]
        if len(d):
            cx, cy = (d[:, 2] + d[:, 4]) / 2, (d[:, 3] + d[:, 5]) / 2
            ins = (cx >= float(t['x1'])) & (cx < float(t['x2'])) & (cy >= float(t['y1'])) & (cy < float(t['y2']))
            if ins.any():
                s[i] = d[ins, 1].max()
    return s


def tile_metrics(tiles, s, thr):
    y = np.array([int(t['label']) for t in tiles])
    im = np.array([t['image'] for t in tiles])
    groups = {k: np.nonzero(im == k)[0] for k in np.unique(im)}
    gk = list(groups)
    bauc, bf1 = [], []
    for _ in range(NBOOT):
        idx = np.concatenate([groups[k] for k in rng.choice(gk, len(gk))])
        bauc.append(roc_auc_score(y[idx], s[idx]))
        bf1.append(f1_score(y[idx], (s[idx] >= thr).astype(int), average='macro'))
    return dict(auc=roc_auc_score(y, s), f1_macro=f1_score(y, (s >= thr).astype(int), average='macro'),
                auc_ci=ci(bauc), f1_macro_ci=ci(bf1), n=len(y), n_defective=int(y.sum()))


res = {'S1_input_size': [], 'S2_eldds_12x12': []}
tiles6 = [t for t in csv.DictReader(open(OUT / 'tiles.csv'))]
for m, ds, img in (('p1_yolov5s', 'eldds', 640), ('p1_gbh', 'eldds', 640), ('p2_v5s', 'pv', 960),
                   ('p2_improved', 'pv', 960), ('p2_v8s', 'pv', 960)):
    tt = [t for t in tiles6 if t['ds'] == ds]
    for label, preds in (('own size', load_preds(m, ds)), (f'@{img}', preds_from(SENS / f'{m}__{ds}@{img}.csv', ds))):
        r = agnostic(m, ds, preds)
        r.update({f'tile_{k}': v for k, v in tile_metrics(tt, tile_scores(tt, defect_preds(m, preds)), DET_THR).items()})
        res['S1_input_size'].append(dict(model=m, ds=ds, size=label, **r))
        print(f'S1 {m:12s} {ds:6s} {label:9s} mAP50 {r["mAP50"]:.3f} [{r["mAP50_ci"][0]:.3f}, {r["mAP50_ci"][1]:.3f}] '
              f'tile AUC {r["tile_auc"]:.3f} F1m {r["tile_f1_macro"]:.3f}', flush=True)

t12 = [t for t in csv.DictReader(open(OUT / 'tiles_eldds12.csv')) if t['ds'] == 'eldds']
cnn12 = {(t['image'], t['r'], t['c']): float(t['score']) for t in csv.DictReader(open(OUT / 'preds/p3_cnn__tiles_eldds12.csv'))
         if t['ds'] == 'eldds'}
for m in MODELS:
    if m == 'p3_cnn':
        s, thr = np.array([cnn12[(t['image'], t['r'], t['c'])] for t in t12]), CNN_THR
    else:
        s, thr = tile_scores(t12, defect_preds(m, load_preds(m, 'eldds'))), DET_THR
    r = tile_metrics(t12, s, thr)
    res['S2_eldds_12x12'].append(dict(model=m, **r))
    print(f'S2 {m:12s} eldds 12x12 AUC {r["auc"]:.3f} [{r["auc_ci"][0]:.3f}, {r["auc_ci"][1]:.3f}] F1m {r["f1_macro"]:.3f}',
          flush=True)
json.dump(res, open(OUT / 'sensitivity.json', 'w'), indent=1)

L = ['### S1: detectors on the foreign dataset at that dataset\'s input size\n',
     '| Model | Dataset | input | class-agnostic mAP@.5 [95% CI] | mAP@.5:.95 | tile AUC [95% CI] | tile macro-F1 |',
     '|---|---|---|---|---|---|---|']
for r in res['S1_input_size']:
    L.append(f'| {MODELS[r["model"]][0]} | {r["ds"]} | {r["size"]} | {r["mAP50"]:.3f} [{r["mAP50_ci"][0]:.3f}, '
             f'{r["mAP50_ci"][1]:.3f}] | {r["mAP"]:.3f} | {r["tile_auc"]:.3f} [{r["tile_auc_ci"][0]:.3f}, '
             f'{r["tile_auc_ci"][1]:.3f}] | {r["tile_f1_macro"]:.3f} |')
L += ['\n### S2: ELDDS tiles 12x12 instead of 6x6\n',
      f'{res["S2_eldds_12x12"][0]["n"]} tiles, {res["S2_eldds_12x12"][0]["n_defective"]} defective.\n',
      '| Model | tile AUC [95% CI] | tile macro-F1 [95% CI] |', '|---|---|---|']
for r in res['S2_eldds_12x12']:
    L.append(f'| {MODELS[r["model"]][0]} | {r["auc"]:.3f} [{r["auc_ci"][0]:.3f}, {r["auc_ci"][1]:.3f}] | '
             f'{r["f1_macro"]:.3f} [{r["f1_macro_ci"][0]:.3f}, {r["f1_macro_ci"][1]:.3f}] |')
(OUT / 'tables_sensitivity.md').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
