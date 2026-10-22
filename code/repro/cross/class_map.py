"""Class-mapped detection: the defect classes of PV-Multi-Defect (P1) and ELDDS1400C5 (P2) mapped onto shared classes
by visual similarity, keeping only the classes that have a counterpart (main protocol: cross_eval.py, class-agnostic).

Mapping (from GT crops, confirmed by what each paper's models predict on the other dataset, fig7):
  crack  PV broken          <-> ELDDS Crack
  line   PV scratch         <-> ELDDS MicroCrack
  dark   PV no_electricity  <-> ELDDS ShortCircuitString + ShortCircuitCell/LowPowerCell
  dropped (no counterpart): PV hot_spot, black_border; ELDDS OtherError (and Examined, not a defect).
Coarse variant: crack + line merged (each paper's models confuse the two on the other dataset).

GT and predictions are relabelled into the shared classes (a model's predictions through its home dataset's map);
dropped classes are removed from both. A shared class that merges several of the model's classes is re-NMSed at the
model's NMS IoU; the others keep the model's own output, so their home AP equals the paper-metric per-class AP. Home
and cross are scored in the same shared space, class-aware, same AP code and bootstrap as
cross_eval.py. Fig 7: for each GT box of the foreign dataset, the class of the best-overlapping foreign detection
(conf >= 0.25, IoU >= 0.1), pooled over each paper's models.

usage (repo root): envs/p3cnn/Scripts/python.exe -I code/repro/cross/class_map.py
Writes results/cross/class_map.json, tables_classmap.md, figs/fig7_class_confusion.png, figs/fig8_mapped_map.png.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from common import DETECTORS, DS_LABEL, MODELS, OUT, defect_gt, defect_preds, load_gt, load_preds  # noqa: E402
from detmetrics import box_iou, ci, det_metrics, det_stats, nms_agnostic  # noqa: E402

NMS_IOU = {m: (0.7 if m == 'p2_v8s' else 0.6) for m in DETECTORS}
DET_THR, NBOOT, CONF_IOU = 0.25, 1000, 0.1
rng = np.random.default_rng(0)
NAMES = {'pv': ['broken', 'hot_spot', 'black_border', 'scratch', 'no_electricity'],
         'eldds': ['Examined', 'ShortCircuitString', 'ShortCircuitCell', 'Crack', 'MicroCrack', 'OtherError']}
SHARED = {'fine': ['crack', 'line', 'dark'], 'coarse': ['crack/line', 'dark']}
MAP = {'pv': {0: 0, 3: 1, 4: 2}, 'eldds': {3: 0, 4: 1, 1: 2, 2: 2}}  # dataset class id -> fine shared id
TO_COARSE = np.array([0, 0, 1])
FIG = OUT / 'figs'


def relabel(a, ds, variant, col=0):
    """Rows of a whose class (column col) is in MAP[ds], with the class replaced by the shared id."""
    keep = np.isin(a[:, col], list(MAP[ds]))
    a = a[keep].copy()
    a[:, col] = [MAP[ds][int(c)] for c in a[:, col]]
    if variant == 'coarse':
        a[:, col] = TO_COARSE[a[:, col].astype(int)]
    return a


def merged_ids(ds, variant):
    """Shared ids that receive more than one class of dataset ds."""
    src = np.array(list(MAP[ds].values()))
    if variant == 'coarse':
        src = TO_COARSE[src]
    ids, n = np.unique(src, return_counts=True)
    return set(ids[n > 1].tolist())


def nms_merged(d, merged, iou_thr, max_det=300):
    """NMS within each merged shared class; other classes keep the model's own NMS output unchanged."""
    out = [nms_agnostic(d[d[:, 0] == c], iou_thr, max_det) if c in merged else d[d[:, 0] == c]
           for c in np.unique(d[:, 0])]
    d = np.concatenate(out) if out else d
    return d[np.argsort(-d[:, 1], kind='stable')][:max_det]


GT = {ds: defect_gt(ds, load_gt(ds)) for ds in ('pv', 'eldds')}
rows = []
for variant in ('fine', 'coarse'):
    for m in DETECTORS:
        home = MODELS[m][1]
        for ds in ('pv', 'eldds'):
            preds = defect_preds(m, load_preds(m, ds))
            keys = sorted(preds)
            mg = merged_ids(home, variant)
            dp = {k: nms_merged(relabel(v, home, variant), mg, NMS_IOU[m]) for k, v in preds.items()}
            dg = {k: relabel(v, ds, variant) for k, v in GT[ds].items()}
            per = det_stats(dp, dg, keys)
            r = det_metrics(per, keys)
            boot = [det_metrics(per, list(rng.choice(keys, len(keys))))['mAP50'] for _ in range(NBOOT)]
            ap = {SHARED[variant][c]: a for c, a in r['ap50_per_class'].items()}
            rows.append(dict(variant=variant, model=m, ds=ds, home=ds == home, mAP50=r['mAP50'], mAP=r['mAP'],
                             P=r['P'], R=r['R'], mAP50_ci=ci(boot), ap50_per_class=ap,
                             n_gt={SHARED[variant][int(c)]: int(n) for c, n in
                                   zip(*np.unique(np.concatenate([v[:, 0] for v in dg.values()]), return_counts=True))}))
            print(f'{variant:6s} {m:12s} {ds:6s} {"home " if ds == home else "cross"} mAP50 {r["mAP50"]:.3f} '
                  f'[{ci(boot)[0]:.3f}, {ci(boot)[1]:.3f}] mAP {r["mAP"]:.3f} '
                  + ' '.join(f'{k} {v:.3f}' for k, v in ap.items()), flush=True)

# foreign-model class confusion (evidence for the mapping), pooled over each paper's models
conf = {}
for paper, home, ds in (('P1', 'pv', 'eldds'), ('P2', 'eldds', 'pv')):
    gcls = [c for c in range(len(NAMES[ds])) if not (ds == 'eldds' and c == 0)]
    pcls = [c for c in range(len(NAMES[home])) if not (home == 'eldds' and c == 0)]
    M = np.zeros((len(gcls), len(pcls) + 1), int)  # last column: no overlapping detection
    models = [m for m in DETECTORS if MODELS[m][1] == home]
    for m in models:
        P = defect_preds(m, load_preds(m, ds))
        for k, g in GT[ds].items():
            d = P[k][P[k][:, 1] >= DET_THR]
            iou = box_iou(g[:, 1:], d[:, 2:]) if len(g) and len(d) else np.zeros((len(g), 0))
            for gi, c in enumerate(g[:, 0].astype(int)):
                j = iou[gi].argmax() if iou.shape[1] else -1
                M[gcls.index(c), pcls.index(int(d[j, 0])) if j >= 0 and iou[gi, j] >= CONF_IOU else -1] += 1
    conf[paper] = dict(gt_ds=ds, models=models, gt_classes=[NAMES[ds][c] for c in gcls],
                       pred_classes=[NAMES[home][c] for c in pcls] + ['none'], counts=M.tolist())

json.dump(dict(mapping={'crack': ['pv:broken', 'eldds:Crack'], 'line': ['pv:scratch', 'eldds:MicroCrack'],
                        'dark': ['pv:no_electricity', 'eldds:ShortCircuitString', 'eldds:ShortCircuitCell'],
                        'dropped': ['pv:hot_spot', 'pv:black_border', 'eldds:OtherError', 'eldds:Examined'],
                        'coarse': {'crack/line': ['crack', 'line'], 'dark': ['dark']}},
               detection=rows, confusion=conf,
               settings=dict(det_threshold=DET_THR, confusion_iou=CONF_IOU, nms_iou=NMS_IOU, n_boot=NBOOT)),
          open(OUT / 'class_map.json', 'w'), indent=1)

# ------------------------------------------------------------------ figures (style as report.py)
INK, INK2, MUTED, BASE, SURF = '#0b0b0b', '#52514e', '#898781', '#c3c2b7', '#fcfcfb'
CMAP = LinearSegmentedColormap.from_list('blues', ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95',
                                                   '#0d366b'])
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'axes.edgecolor': BASE, 'axes.labelcolor': INK2,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.titlecolor': INK, 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.titlesize': 11})


def heat(ax, mat, rowl, coll, txt, boxed=None, vmax=1):
    ax.imshow(mat, cmap=CMAP, vmin=0, vmax=vmax, aspect='auto')
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, txt[i][j], ha='center', va='center', fontsize=9,
                    color='white' if mat[i, j] / vmax > 0.55 else INK,
                    fontweight='bold' if boxed is not None and boxed[i][j] else 'normal')
            if boxed is not None and boxed[i][j]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=2))
    ax.set_xticks(range(len(coll)), coll)
    ax.set_yticks(range(len(rowl)), rowl)
    ax.tick_params(length=0, labelcolor=INK2)
    for s in ax.spines.values():
        s.set_visible(False)


FINE_OF = {n: s for s, ns in (('crack', ['broken', 'Crack']), ('line', ['scratch', 'MicroCrack']),
                              ('dark', ['no_electricity', 'ShortCircuitString', 'ShortCircuitCell'])) for n in ns}
fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.4), gridspec_kw={'wspace': 0.5})
for ax, (paper, c) in zip(axs, conf.items()):
    M = np.array(c['counts'], float)
    nm = len(c['models'])
    frac = M / M.sum(1, keepdims=True)
    txt = [[f'{frac[i, j]:.2f}' if M[i, j] else '·' for j in range(M.shape[1])] for i in range(M.shape[0])]
    boxed = [[FINE_OF.get(g) is not None and FINE_OF.get(g) == FINE_OF.get(p) for p in c['pred_classes']]
             for g in c['gt_classes']]
    heat(ax, frac, [f'{g}  (n={int(M[i].sum()) // nm})' for i, g in enumerate(c['gt_classes'])],
         c['pred_classes'], txt, boxed)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')
    ax.set_xlabel(f'class predicted by the {paper} models (best overlap, conf ≥ {DET_THR})')
    ax.set_title(f'{paper} models ({nm}) on {DS_LABEL[c["gt_ds"]]}: GT class → predicted class', loc='left')
fig.text(0.01, -0.14, f'Fraction of GT boxes (n = GT boxes), pooled over the paper\'s models; overlap = IoU ≥ '
         f'{CONF_IOU}. Boxed cells = pairs mapped to the same shared class.', color=MUTED, fontsize=9)
fig.savefig(FIG / 'fig7_class_confusion.png', dpi=150, bbox_inches='tight')
plt.close(fig)

R = {(r['variant'], r['model'], r['ds']): r for r in rows}
other = lambda m: 'eldds' if MODELS[m][1] == 'pv' else 'pv'  # noqa: E731
cols = SHARED['fine'] + ['mAP@.5']
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={'wspace': 0.05})
for ax, side in zip(axs, ('home', 'cross')):
    vals, txt = [], []
    for m in DETECTORS:
        r = R[('fine', m, MODELS[m][1] if side == 'home' else other(m))]
        v = [r['ap50_per_class'][k] for k in SHARED['fine']] + [r['mAP50']]
        vals.append(v)
        txt.append([f'{x:.3f}' for x in v[:-1]] + [f'{v[-1]:.3f}\n[{r["mAP50_ci"][0]:.2f}, {r["mAP50_ci"][1]:.2f}]'])
    heat(ax, np.array(vals), [MODELS[m][0] for m in DETECTORS] if side == 'home' else [''] * len(DETECTORS), cols, txt,
         [[j == 3 for j in range(4)] for _ in DETECTORS])
    if side == 'cross':
        ax.set_yticks([])
    ax.set_title(('Home dataset' if side == 'home' else "Other paper's dataset") + ': AP@.5 per shared class',
                 loc='left')
fig.savefig(FIG / 'fig8_mapped_map.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------------------ tables
L = ['### Class-mapped detection (shared classes crack / line / dark), AP@.5\n',
     '| Model | Dataset | | crack | line | dark | mAP@.5 [95% CI] | mAP@.5:.95 | P | R |',
     '|---|---|---|---|---|---|---|---|---|---|']
for m in DETECTORS:
    for ds in (MODELS[m][1], other(m)):
        r = R[('fine', m, ds)]
        a = r['ap50_per_class']
        L.append(f'| {MODELS[m][0]} | {DS_LABEL[ds]} | {"home" if r["home"] else "cross"} | {a["crack"]:.3f} | '
                 f'{a["line"]:.3f} | {a["dark"]:.3f} | {r["mAP50"]:.3f} [{r["mAP50_ci"][0]:.3f}, {r["mAP50_ci"][1]:.3f}] '
                 f'| {r["mAP"]:.3f} | {r["P"]:.3f} | {r["R"]:.3f} |')
n = {ds: R[('fine', DETECTORS[0], ds)]['n_gt'] for ds in ('pv', 'eldds')}
L.append('\nGT boxes kept: ' + '; '.join(f'{DS_LABEL[ds]} ' + ', '.join(f'{k} {v}' for k, v in n[ds].items())
                                         for ds in n) + '.')
L += ['\n### Coarse variant (crack + line merged), AP@.5\n',
      '| Model | Dataset | | crack/line | dark | mAP@.5 [95% CI] |', '|---|---|---|---|---|---|']
for m in DETECTORS:
    for ds in (MODELS[m][1], other(m)):
        r = R[('coarse', m, ds)]
        a = r['ap50_per_class']
        L.append(f'| {MODELS[m][0]} | {DS_LABEL[ds]} | {"home" if r["home"] else "cross"} | {a["crack/line"]:.3f} | '
                 f'{a["dark"]:.3f} | {r["mAP50"]:.3f} [{r["mAP50_ci"][0]:.3f}, {r["mAP50_ci"][1]:.3f}] |')
L += ['\n### Foreign-model class confusion (fraction of GT boxes; IoU >= 0.1, conf >= 0.25)\n']
for paper, c in conf.items():
    M = np.array(c['counts'], float)
    L += [f'**{paper} models ({", ".join(MODELS[m][0] for m in c["models"])}) on {DS_LABEL[c["gt_ds"]]}**\n',
          '| GT \\ predicted | GT boxes | ' + ' | '.join(c['pred_classes']) + ' |',
          '|---|---|' + '---|' * len(c['pred_classes'])]
    for i, g in enumerate(c['gt_classes']):
        L.append(f'| {g} | {int(M[i].sum()) // len(c["models"])} | ' + ' | '.join(f'{x / M[i].sum():.2f}' for x in M[i]) + ' |')
    L.append('')
(OUT / 'tables_classmap.md').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
