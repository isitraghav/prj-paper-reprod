"""P3 claims that only R1 tested (SVM feature/grid study, Figs. 6-7; training-set size, Fig. 10), re-scored with our
evaluator from R1's saved test-set outputs (R1 test split; outputs aligned with cnn_full.npz 'te', checked).

Scores: confidence-weighted macro-F1 (Fig. 6 metric; SVM decision >= 0) and weighted ROC AUC (Fig. 7, Fig. 10).
R1 pipeline: OpenCV 3.4.2 features, VLAD, PCA whitening, LIBLINEAR-objective linear SVM / LIBSVM RBF (R1 report).

usage (repo root): envs/p3cnn/Scripts/python.exe -I code/repro/replication/r1_p3_svm.py <R1 folder>
-> results/replication_r1/p3_svm.json, p3_svm.md
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/repro/p3'))
from elpv_common import confidence_weight  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
W = Path(sys.argv[1]) / 'final_work'
rows = [l.split() for l in (ROOT / 'datasets/raw/elpv-dataset/src/elpv_dataset/data/labels.csv').read_text().splitlines() if l.strip()]
p = np.array([float(r[1]) for r in rows])
typ = np.array([r[2] for r in rows])
te = np.load(W / 'deitsch_cnn/cnn_full.npz', allow_pickle=False)['te']
y, w, t = (p[te] > 0).astype(int), confidence_weight(p[te]), typ[te]


def score(s, thr=0.0, sub=None):
    m = np.ones(len(y), bool) if sub is None else t == sub
    return {'f1_w': float(f1_score(y[m], (s[m] >= thr).astype(int), average='macro', sample_weight=w[m])),
            'auc_w': float(roc_auc_score(y[m], s[m], sample_weight=w[m]))}


out = {'dense': [], 'keypoint': {}, 'subsets': {}}
for f in sorted(glob.glob(str(W / 'deitsch_svm/dense_*_scores.npz'))):
    name = os.path.basename(f)[:-len('_scores.npz')]
    if name.endswith(('_lin', '_rbf')):
        continue  # later single-kernel re-runs without saved test indices; the full 4-variant files are used
    z = np.load(f, allow_pickle=False)
    assert np.array_equal(z['te'], te)
    _, desc, grid, *mask = name.split('_')
    for k in ('linear_w', 'linear_nw', 'rbf_w', 'rbf_nw'):
        out['dense'].append({'desc': desc, 'grid': int(grid), 'mask': bool(mask), 'kernel': k.split('_')[0],
                             'weighted': k.endswith('_w'), **score(z[k])})
KP = ['kaze_vgg', 'kaze_sift', 'kaze_surf', 'kaze', 'agast_vgg', 'agast_sift', 'agast_surf', 'hog', 'phow']
for name in KP:
    z = np.load(W / f'deitsch_svm/{name}_scores.npz', allow_pickle=False)
    if 'te' in z.files:
        assert np.array_equal(z['te'], te)
    out['keypoint'][name] = {s or 'all': score(z['linear_w'], sub=s)['auc_w'] for s in (None, 'mono', 'poly')}
for desc, g in (('sift', 60), ('surf', 70), ('vgg', 65)):  # the paper's best dense grids, as in its Fig. 7
    z = np.load(W / f'deitsch_svm/dense_{desc}_{g}_scores.npz', allow_pickle=False)
    out['keypoint'][f'dense_{desc}_{g}'] = {s or 'all': score(z['linear_w'], sub=s)['auc_w'] for s in (None, 'mono', 'poly')}
for model, pattern in (('svm', 'deitsch_svm_subsets/f*_s*.json'), ('cnn', 'deitsch_cnn_subsets/cnn_f*_s*.json')):
    by = {}
    for f in glob.glob(str(W / pattern)):
        d = json.load(open(f))
        s = np.asarray(d.get('scores', d.get('pred')), float)
        if s.ndim != 1 or len(s) != len(te):
            continue
        by.setdefault(float(d.get('frac', d.get('train_frac'))), []).append(score(s, thr=0.5)['auc_w'])
    out['subsets'][model] = {str(k): {'n': len(v), 'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
                             for k, v in sorted(by.items())}
full_svm = np.load(W / 'deitsch_svm/kaze_vgg_cv_scores.npz', allow_pickle=False)['linear_w']
out['subsets']['svm']['1.0'] = {'n': 1, 'mean': score(full_svm)['auc_w'], 'sd': 0.0}
out['subsets']['cnn']['1.0'] = {'n': 1, 'mean': score(np.load(W / 'deitsch_cnn/cnn_full.npz')['pred'], 0.5)['auc_w'], 'sd': 0.0}

# ---------------------------------------------------------------- claim checks
D = out['dense']
best = {}
for desc in ('surf', 'sift', 'vgg'):
    r = max((x for x in D if x['desc'] == desc), key=lambda x: x['f1_w'])
    curve = {}
    for x in D:
        if x['desc'] == desc:
            curve[x['grid']] = max(curve.get(x['grid'], 0), x['f1_w'])
    g = sorted(curve)
    best[desc] = dict(r, rho_grid=float(spearmanr(g, [curve[k] for k in g]).correlation))
pairs = lambda key: [(a, b) for a in D for b in D if a['desc'] == b['desc'] and a['grid'] == b['grid'] and  # noqa: E731
                     all(a[k] == b[k] for k in ('mask', 'kernel', 'weighted') if k != key) and a[key] and not b[key]]
wt = pairs('weighted')
mk = pairs('mask')
mk_vgg = [(a, b) for a, b in mk if a['desc'] == 'vgg']
rank = sorted(out['keypoint'], key=lambda k: -out['keypoint'][k]['all'])
out['checks'] = {
    'best_dense': best,
    'weighting_helps_fraction': float(np.mean([a['f1_w'] > b['f1_w'] for a, b in wt])),
    'masking_helps_fraction_all': float(np.mean([a['f1_w'] > b['f1_w'] for a, b in mk])),
    'masking_helps_fraction_vgg': float(np.mean([a['f1_w'] > b['f1_w'] for a, b in mk_vgg])),
    'masking_gain_best_vgg': max(x['f1_w'] for x in D if x['desc'] == 'vgg' and x['mask']) - max(x['f1_w'] for x in D if x['desc'] == 'vgg' and not x['mask']),
    'fig7_rank': rank,
    'best_keypoint_vs_best_dense_auc': max(v['all'] for k, v in out['keypoint'].items() if not k.startswith('dense')) - max(v['all'] for k, v in out['keypoint'].items() if k.startswith('dense')),
}
RES = ROOT / 'results/replication_r1'
json.dump(out, open(RES / 'p3_svm.json', 'w'), indent=1)

c = out['checks']
PAPER_BEST = {'surf': '70 × 70, RBF, weighted, no mask', 'sift': '60 × 60, linear, weighted, no mask', 'vgg': '65 × 65, linear, weighted, masked'}
L = ['# P3 claims tested only by R1, re-scored with our evaluator', '', 'Generated by `code/repro/replication/r1_p3_svm.py`. '
     'R1 test split (656 cells, Table 3 counts). Confidence-weighted macro-F1 / AUC.', '',
     '## Fig. 6: dense sampling (grid 5–75, mask, kernel, weighting)', '',
     '| Descriptor | Paper best | R1 best (macro-F1_w) | Trend with grid density (Spearman ρ of best F1 vs grid) |', '|---|---|---|---|']
for d, r in c['best_dense'].items():
    L.append(f"| {d.upper()} | {PAPER_BEST[d]} | {r['grid']} × {r['grid']}, {'RBF' if r['kernel'] == 'rbf' else 'linear'}, "
             f"{'weighted' if r['weighted'] else 'unweighted'}, {'masked' if r['mask'] else 'no mask'} ({100 * r['f1_w']:.1f}) | {r['rho_grid']:+.2f} |")
L += ['', f"- Weighting helps (weighted > unweighted, same other options): **{100 * c['weighting_helps_fraction']:.0f} %** of {len(wt)} pairs (paper: \"in most cases\").",
      f"- Masking helps: {100 * c['masking_helps_fraction_all']:.0f} % of pairs overall, {100 * c['masking_helps_fraction_vgg']:.0f} % for VGG; "
      f"best masked − best unmasked VGG F1 = {100 * c['masking_gain_best_vgg']:+.1f} points (paper: VGG gains \"almost two percent\").", '',
      '## Fig. 7: feature combinations (linear SVM, weighted, no mask), weighted AUC', '',
      '| Rank | Features | all | mono | poly |', '|---|---|---|---|---|']
for i, k in enumerate(c['fig7_rank'], 1):
    v = out['keypoint'][k]
    L.append(f"| {i} | {k} | {100 * v['all']:.2f} | {100 * v['mono']:.2f} | {100 * v['poly']:.2f} |")
L += ['', f"Paper: KAZE/VGG first (88.51), KAZE/SIFT second (87.22); keypoints beat dense sampling. R1: best keypoint − best dense AUC = "
      f"{100 * c['best_keypoint_vs_best_dense_auc']:+.2f} points.", '',
      '## Fig. 10: training-set size (weighted AUC, mean ± sd over random stratified subsets)', '',
      '| Fraction | SVM (n) | CNN (n) |', '|---|---|---|']
for fr in sorted(set(out['subsets']['svm']) | set(out['subsets']['cnn']), key=float):
    s, n = out['subsets']['svm'].get(fr), out['subsets']['cnn'].get(fr)
    fmt = lambda x: f"{100 * x['mean']:.1f} ± {100 * x['sd']:.1f} ({x['n']})" if x else '—'  # noqa: E731
    L.append(f'| {float(fr):.0%} | {fmt(s)} | {fmt(n)} |')
(RES / 'p3_svm.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
print('\n'.join(L))
