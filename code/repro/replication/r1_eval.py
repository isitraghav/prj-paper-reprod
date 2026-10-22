"""Replication R1 (earlier RTX 3050 run of P1 and P3, outside the repo) vs our main run R2, scored with our evaluators.

P1: R1's own home numbers (its yolov5 test.py JSONs, R1's val split, last.pt) and the training options behind them;
    every R1 model and our R2 models on all three test sets with the cross pipeline's code (code/repro/cross), plus
    paired bootstrap tests. On PV-Multi-Defect the R1 numbers are leaky: R1's split was not saved, and R1 trained on
    about 80 % of our val images. ELDDS and ELPV are foreign to both runs.
P3: R1's saved test-set outputs (CNN probabilities, linear-SVM decision values) re-scored with
    code/repro/p3/elpv_common.metrics on R1's own test split (saved as indices into labels.csv).

usage (repo root, after run_r1_dumps.sh):
  envs/p3cnn/Scripts/python.exe -I code/repro/replication/r1_eval.py <R1 folder>
-> results/replication_r1/summary.json, summary.md
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/repro/cross'))
sys.path.insert(0, str(ROOT / 'code/repro/p3'))
from common import OUT, defect_gt, image_list, load_gt, stem  # noqa: E402
from detmetrics import ci, det_metrics, det_stats, nms_agnostic  # noqa: E402
from elpv_common import confidence_weight, metrics  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
R1 = Path(sys.argv[1])
RUNS = R1 / 'elec_paper/GBH-YOLOv5/runs'
RES = ROOT / 'results/replication_r1'
NBOOT = 1000
rng = np.random.default_rng(0)

# model id -> (label, prediction csv prefix, R1 eval json, R1 train dir)
P1 = {
    'r2_yolov5s': ('YOLOv5s (R2)', OUT / 'preds/p1_yolov5s', None, None),
    'r1_yolov5s': ('YOLOv5s (R1)', RES / 'preds/r1_abl_yolov5s', 'yolov5s', 'abl_yolov5s'),
    'r2_yolov5_1': ('YOLOv5-1 (R2)', RES / 'preds/r2_yolov5_1', None, None),
    'r1_yolov5_1': ('YOLOv5-1 (R1)', RES / 'preds/r1_abl_yolov5s_1', 'yolov5s_1', 'abl_yolov5s_1'),
    'r1_yolov5_2': ('YOLOv5-2 (R1)', RES / 'preds/r1_abl_yolov5s_2', 'yolov5s_2', 'abl_yolov5s_2'),
    'r2_gbh': ('GBH-YOLOv5 (R2)', OUT / 'preds/p1_gbh', None, None),
    'r1_gbh': ('GBH-YOLOv5 (R1)', RES / 'preds/r1_pv_gbh_b16', 'gbh', 'pv_gbh_b16'),
}
PAIRS = [('r1_yolov5s', 'r2_yolov5s'), ('r1_yolov5_1', 'r2_yolov5_1'), ('r1_gbh', 'r2_gbh'),
         ('r1_gbh', 'r1_yolov5s'), ('r1_yolov5_1', 'r1_yolov5s'), ('r1_yolov5_2', 'r1_yolov5s'), ('r2_gbh', 'r2_yolov5s')]


def load(prefix, ds):
    out = {stem(p): [] for p in image_list(ds)}
    for r in csv.DictReader(open(f'{prefix}__{ds}.csv')):
        out[r['image']].append([float(r[k]) for k in ('cls', 'conf', 'x1', 'y1', 'x2', 'y2')])
    return {k: np.array(v, float).reshape(-1, 6) for k, v in out.items()}


def opt_value(f, key):
    lines = open(f).read().splitlines()
    for i, l in enumerate(lines):
        if l.startswith(key + ':'):
            v = l.split(':', 1)[1].strip()
            return v if v else lines[i + 1].lstrip('- ').strip()


GT = {ds: load_gt(ds) for ds in ('pv', 'eldds')}
tiles = list(csv.DictReader(open(OUT / 'tiles.csv')))
summary = {'p1': {}, 'p1_paired': [], 'p3': {}}
per_model = {}
for m, (label, prefix, ev, tr) in P1.items():
    P = {ds: load(prefix, ds) for ds in ('pv', 'eldds', 'elpv')}
    r = {'label': label}
    if ev:
        e = json.load(open(RUNS / f'eval/{ev}.json'))
        o = RUNS / f'train/{tr}/opt.yaml'
        r['r1_home'] = {'mAP50': e['mAP50'], 'P': e['P'], 'R': e['R'], 'mAP50_95': e['mAP50_95'],
                        'ap50_per_class': {k: v['AP50'] for k, v in e['per_class'].items()}, 'weights': e['weights'],
                        'img': opt_value(o, 'img_size'), 'batch': opt_value(o, 'batch_size'),
                        'epochs': opt_value(o, 'epochs')}
    keys = sorted(P['pv'])
    r['pv_home_mAP50'] = float(det_metrics(det_stats(P['pv'], GT['pv'], keys), keys)['mAP50'])
    stats = {}
    for ds in ('pv', 'eldds'):
        keys = sorted(P[ds])
        dp = {k: nms_agnostic(np.c_[np.zeros(len(v)), v[:, 1:]], 0.6) for k, v in P[ds].items()}
        dg = {k: np.c_[np.zeros(len(v)), v[:, 1:]] for k, v in defect_gt(ds, GT[ds]).items()}
        stats[ds] = per = det_stats(dp, dg, keys)
        r[f'{ds}_agn_mAP50'] = float(det_metrics(per, keys)['mAP50'])
    tile = {}
    for ds in ('pv', 'eldds', 'elpv'):
        T = [t for t in tiles if t['ds'] == ds]
        s = np.zeros(len(T))
        for i, t in enumerate(T):
            d = P[ds][t['image']]
            if len(d) and ds != 'elpv':
                cx, cy = (d[:, 2] + d[:, 4]) / 2, (d[:, 3] + d[:, 5]) / 2
                d = d[(cx >= float(t['x1'])) & (cx < float(t['x2'])) & (cy >= float(t['y1'])) & (cy < float(t['y2']))]
            if len(d):
                s[i] = d[:, 1].max()
        y, img = np.array([int(t['label']) for t in T]), np.array([t['image'] for t in T])
        tile[ds] = (y, s, img)
        r[f'{ds}_tile_auc'] = float(roc_auc_score(y, s))
    per_model[m] = (stats, tile)
    summary['p1'][m] = r

# bootstrap CIs (per model) and paired differences, on the same image resamples
eld_keys = sorted(stem(p) for p in image_list('eldds'))
idx = {ds: {k: np.where(per_model['r2_yolov5s'][1][ds][2] == k)[0] for k in np.unique(per_model['r2_yolov5s'][1][ds][2])}
       for ds in ('eldds', 'elpv')}
boot_det = [list(rng.choice(eld_keys, len(eld_keys))) for _ in range(NBOOT)]
boot_tile = {ds: [np.concatenate([idx[ds][k] for k in rng.choice(list(idx[ds]), len(idx[ds]))]) for _ in range(NBOOT)]
             for ds in ('eldds', 'elpv')}


def boot_auc(tile, j):
    y, s, _ = tile
    return roc_auc_score(y[j], s[j])


for m in P1:
    stats, tile = per_model[m]
    summary['p1'][m]['eldds_agn_mAP50_ci'] = ci([det_metrics(stats['eldds'], b)['mAP50'] for b in boot_det])
    for ds in ('eldds', 'elpv'):
        summary['p1'][m][f'{ds}_tile_auc_ci'] = ci([boot_auc(tile[ds], j) for j in boot_tile[ds]])
for a, b in PAIRS:
    (sa, ta), (sb, tb) = per_model[a], per_model[b]
    row = {'a': a, 'b': b,
           'eldds_agn_mAP50': summary['p1'][a]['eldds_agn_mAP50'] - summary['p1'][b]['eldds_agn_mAP50'],
           'eldds_agn_mAP50_ci': ci([det_metrics(sa['eldds'], k)['mAP50'] - det_metrics(sb['eldds'], k)['mAP50'] for k in boot_det])}
    for ds in ('eldds', 'elpv'):
        row[f'{ds}_tile_auc'] = summary['p1'][a][f'{ds}_tile_auc'] - summary['p1'][b][f'{ds}_tile_auc']
        row[f'{ds}_tile_auc_ci'] = ci([boot_auc(ta[ds], j) - boot_auc(tb[ds], j) for j in boot_tile[ds]])
    summary['p1_paired'].append(row)

# ---------------------------------------------------------------- P3
rows = [l.split() for l in (ROOT / 'datasets/raw/elpv-dataset/src/elpv_dataset/data/labels.csv').read_text().splitlines() if l.strip()]
path = np.array([r[0] for r in rows])
p = np.array([float(r[1]) for r in rows])
typ = np.array([r[2] for r in rows])
our_test = set((ROOT / 'datasets/splits/elpv/test.txt').read_text().split())
cnn = np.load(R1 / 'final_work/deitsch_cnn/cnn_full.npz', allow_pickle=False)
svm = np.load(R1 / 'final_work/deitsch_svm/kaze_vgg_cv_scores.npz', allow_pickle=False)
r1_sum = json.load(open(R1 / 'final/results/paper2_deitsch2019/summary.json'))
rbf = json.load(open(R1 / 'final_work/deitsch_svm/kaze_vgg_cv_rbf.json'))
for name, te, s, thr in (('cnn', cnn['te'], cnn['pred'], 0.5), ('svm_linear', svm['te'], svm['linear_w'], 0.0)):
    m = metrics(p[te], s, typ[te], threshold=thr)
    y, yh = (p[te] > 0).astype(int), (s >= thr).astype(int)
    summary['p3'][name] = {'n': len(te), 'overlap_with_our_test': len(set(path[te]) & our_test),
                           'acc_w': m['all']['acc_weighted'], 'f1_macro_w': m['all']['f1_macro_weighted'],
                           'f1_support_w': f1_score(y, yh, average='weighted', sample_weight=confidence_weight(p[te])),
                           'auc_w': m['all']['auc_weighted'],
                           'auc_w_mono': m['mono']['auc_weighted'], 'auc_w_poly': m['poly']['auc_weighted']}
summary['p3']['same_split_cnn_svm'] = bool(np.array_equal(np.sort(cnn['te']), np.sort(svm['te'])))
summary['p3']['r1_reported'] = {'cnn_acc_f1_auc': r1_sum['cnn']['Overall'], 'svm_rbf': rbf['all'], 'svm_rbf_best': rbf['best']}

RES.mkdir(parents=True, exist_ok=True)
json.dump(summary, open(RES / 'summary.json', 'w'), indent=1)

# ---------------------------------------------------------------- markdown
f3 = lambda v: f'{v:.3f}'  # noqa: E731
fci = lambda c: f'[{c[0]:.3f}, {c[1]:.3f}]'  # noqa: E731
L = ['# Replication R1 vs main run R2', '', 'Generated by `code/repro/replication/r1_eval.py`. R1 = the earlier RTX 3050 run '
     '(600 px), R2 = our main run (960 px, `models/`).', '',
     '## P1: home (each run on its own val split)', '',
     '| Model | R1 mAP@.5 (R1 val 221, last.pt) | R1 settings (img / batch / epochs) | R2 on our val 222 (this evaluator) | R1 on our val (leaky) |',
     '|---|---|---|---|---|']
for m, r in summary['p1'].items():
    h = r.get('r1_home')
    sett = f"{h['img']} / {h['batch']} / {h['epochs']}" if h else ''
    L.append(f"| {r['label']} | {f3(h['mAP50']) if h else ''} | {sett} | "
             f"{f3(r['pv_home_mAP50']) if m.startswith('r2') else ''} | {f3(r['pv_home_mAP50']) if m.startswith('r1') else ''} |")
L += ['', '## P1: foreign datasets (clean for both runs)', '',
      '| Model | ELDDS class-agnostic mAP@.5 [95 % CI] | ELDDS tile AUC [CI] | ELPV tile AUC [CI] |', '|---|---|---|---|']
for m, r in summary['p1'].items():
    L.append(f"| {r['label']} | {f3(r['eldds_agn_mAP50'])} {fci(r['eldds_agn_mAP50_ci'])} | "
             f"{f3(r['eldds_tile_auc'])} {fci(r['eldds_tile_auc_ci'])} | {f3(r['elpv_tile_auc'])} {fci(r['elpv_tile_auc_ci'])} |")
L += ['', '## P1: paired differences (same bootstrap resamples)', '',
      '| A − B | ELDDS mAP@.5 | ELDDS tile AUC | ELPV tile AUC |', '|---|---|---|---|']
lab = {m: r['label'] for m, r in summary['p1'].items()}
for row in summary['p1_paired']:
    L.append(f"| {lab[row['a']]} − {lab[row['b']]} | {row['eldds_agn_mAP50']:+.3f} {fci(row['eldds_agn_mAP50_ci'])} | "
             f"{row['eldds_tile_auc']:+.3f} {fci(row['eldds_tile_auc_ci'])} | {row['elpv_tile_auc']:+.3f} {fci(row['elpv_tile_auc_ci'])} |")
c, s = summary['p3']['cnn'], summary['p3']['svm_linear']
rep = summary['p3']['r1_reported']
L += ['', '## P3: R1 outputs re-scored with our evaluator (R1 test split)', '',
      f"R1's test split has the paper's Table 3 counts; {c['overlap_with_our_test']} of its 656 cells are in our test split.", '',
      '| Model | acc_w | macro-F1_w | support-F1_w | AUC_w | AUC_w mono / poly | R1 report (acc / F1 / AUC) |', '|---|---|---|---|---|---|---|',
      f"| CNN (`cnn_vgg19_paper.pt` outputs) | {100 * c['acc_w']:.2f} | {100 * c['f1_macro_w']:.2f} | {100 * c['f1_support_w']:.2f} | "
      f"{100 * c['auc_w']:.2f} | {100 * c['auc_w_mono']:.2f} / {100 * c['auc_w_poly']:.2f} | "
      + ' / '.join(f'{100 * v:.2f}' for v in rep['cnn_acc_f1_auc']) + ' |',
      f"| SVM KAZE/VGG linear | {100 * s['acc_w']:.2f} | {100 * s['f1_macro_w']:.2f} | {100 * s['f1_support_w']:.2f} | "
      f"{100 * s['auc_w']:.2f} | {100 * s['auc_w_mono']:.2f} / {100 * s['auc_w_poly']:.2f} | |",
      f"| SVM KAZE/VGG RBF (no outputs saved) | | | | | | {100 * rep['svm_rbf']['acc']:.2f} / {100 * rep['svm_rbf']['f1']:.2f} / {100 * rep['svm_rbf']['auc']:.2f} |"]
(RES / 'summary.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
print('\n'.join(L))
