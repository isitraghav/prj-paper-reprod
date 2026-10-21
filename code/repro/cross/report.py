"""Figures + markdown tables for the cross-dataset comparison (reads results/cross/metrics.json, curves.json, preds).

usage (repo root): envs/p3cnn/Scripts/python.exe -I code/repro/cross/report.py
Writes results/cross/figs/*.png and results/cross/tables.md.
"""
import csv
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
from PIL import Image  # noqa: E402

from common import (COLOR, DATASETS, DETECTORS, DS_LABEL, MODELS, OUT, defect_gt, defect_preds, image_list,  # noqa: E402
                    load_gt, load_preds, stem)

FIG = OUT / 'figs'
FIG.mkdir(exist_ok=True)
M = json.load(open(OUT / 'metrics.json'))
C = json.load(open(OUT / 'curves.json'))
INK, INK2, MUTED, GRID, BASE, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
BLUES = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf',
         '#1c5cab', '#184f95', '#104281', '#0d366b']
CMAP = LinearSegmentedColormap.from_list('blues', BLUES)
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'axes.edgecolor': BASE, 'axes.labelcolor': INK2,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.titlecolor': INK, 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False, 'axes.titlesize': 11})
LAB = {m: v[0] for m, v in MODELS.items()}
HOME = {m: v[1] for m, v in MODELS.items()}
DET = {(r['model'], r['ds']): r for r in M['detection'] if r['mode'] == 'agnostic'}
DET_HOME = {r['model']: r for r in M['detection'] if r['mode'] == 'home'}
TIL = {(r['model'], r['ds']): r for r in M['tiles']}
ELDDS_NAMES = ['Examined', 'ShortCircuitString', 'ShortCircuitCell', 'Crack', 'MicroCrack', 'OtherError']
WRAP = {'ShortCircuitString': 'ShortCircuit-\nString', 'ShortCircuitCell': 'ShortCircuit-\nCell',
        'no_electricity': 'no_\nelectricity', 'black_border': 'black_\nborder'}
PV_NAMES = ['broken', 'hot_spot', 'black_border', 'scratch', 'no_electricity']


def heat(ax, mat, rows, cols, fmt, home=None, ci=None, vmin=0, vmax=1, title=''):
    ax.imshow(mat, cmap=CMAP, vmin=vmin, vmax=vmax, aspect='auto')
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, 'n/a', ha='center', va='center', color=MUTED, fontsize=9)
                continue
            dark = (v - vmin) / (vmax - vmin) > 0.55
            s = fmt.format(v) + (f'\n[{ci[i][j][0]:.2f}, {ci[i][j][1]:.2f}]' if ci is not None and ci[i][j] else '')
            ax.text(j, i, s, ha='center', va='center', color='white' if dark else INK, fontsize=9,
                    fontweight='bold' if home is not None and home[i][j] else 'normal')
            if home is not None and home[i][j]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=2))
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0, labelcolor=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc='left')


# ------------------------------------------------------------------ fig 1: detection matrix
dets = DETECTORS
mat = np.array([[DET[(m, d)]['mAP50'] for d in ('pv', 'eldds')] for m in dets])
mat95 = np.array([[DET[(m, d)]['mAP'] for d in ('pv', 'eldds')] for m in dets])
ci = [[DET[(m, d)]['mAP50_ci'] for d in ('pv', 'eldds')] for m in dets]
home = [[HOME[m] == d for d in ('pv', 'eldds')] for m in dets]
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={'wspace': 0.05})
heat(axs[0], mat, [LAB[m] for m in dets], [DS_LABEL[d] for d in ('pv', 'eldds')], '{:.3f}', home, ci,
     title='Class-agnostic mAP@.5  [95% bootstrap CI]')
heat(axs[1], mat95, [''] * len(dets), [DS_LABEL[d] for d in ('pv', 'eldds')], '{:.3f}', home,
     title='Class-agnostic mAP@.5:.95')
axs[1].set_yticks([])
fig.text(0.01, 0.01, 'Outlined = model evaluated on its own paper\'s dataset. All defect classes merged into one; '
         'ELDDS "Examined" module boxes excluded.', color=MUTED, fontsize=8)
fig.savefig(FIG / 'fig1_detection_matrix.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------------------ fig 2: tile matrix
models = list(MODELS)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={'wspace': 0.05})
for ax, key, title, lab in ((axs[0], 'auc', 'Defective-vs-functional tile AUC  [95% CI]', True),
                            (axs[1], 'f1_macro', 'Tile macro-F1 at fixed threshold  [95% CI]', False)):
    mt = np.array([[TIL[(m, d)][key] for d in DATASETS] for m in models])
    cit = [[TIL[(m, d)][key + '_ci'] for d in DATASETS] for m in models]
    hm = [[HOME[m] == d for d in DATASETS] for m in models]
    heat(ax, mt, [LAB[m] for m in models] if lab else [''] * len(models), [DS_LABEL[d] for d in DATASETS], '{:.3f}',
         hm, cit, vmin=0.3, vmax=1, title=title)
    if not lab:
        ax.set_yticks([])
fig.text(0.01, 0.0, 'Tiles: PV 4x4 grid, ELDDS 6x6 grid over the module, ELPV = one cell. Thresholds: CNN 0.5, '
         'detectors conf 0.25 (pre-declared). Outlined = home dataset.', color=MUTED, fontsize=8)
fig.savefig(FIG / 'fig2_tile_matrix.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------------------ fig 3: PR curves (class-agnostic, IoU .5)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, d in zip(axs, ('pv', 'eldds')):
    for m in dets:
        r, p = C['pr'][f'{m}__{d}']
        ls = '-' if HOME[m] == d else '--'
        ax.plot(r, p, ls, color=COLOR[m], lw=2, label=LAB[m])
    ax.set_title(f'{DS_LABEL[d]}: precision-recall, defect @ IoU 0.5', loc='left')
    ax.set_xlabel('Recall')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(color=GRID, lw=0.6)
axs[0].set_ylabel('Precision')
fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.06), fontsize=8.5,
           labelcolor=INK2, handlelength=2.5)
fig.text(0.01, -0.1, 'Solid = home dataset, dashed = foreign dataset. Values: Fig. 1.', color=MUTED, fontsize=8)
fig.savefig(FIG / 'fig3_pr_curves.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------------------ fig 4: ROC curves (tiles)
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
for ax, d in zip(axs, DATASETS):
    ax.plot([0, 1], [0, 1], color=BASE, lw=1)
    for m in models:
        f, t = C['roc'][f'{m}__{d}']
        ax.plot(f, t, '-' if HOME[m] == d else '--', color=COLOR[m], lw=2, label=LAB[m])
    ax.set_title(f'{DS_LABEL[d]}: tile ROC', loc='left')
    ax.set_xlabel('False positive rate')
    ax.grid(color=GRID, lw=0.6)
axs[0].set_ylabel('True positive rate')
fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.06), fontsize=8.5,
           labelcolor=INK2, handlelength=2.5)
fig.text(0.01, -0.1, 'Solid = home dataset (its own paper), dashed = foreign. Values: Fig. 2.', color=MUTED,
         fontsize=8)
fig.savefig(FIG / 'fig4_tile_roc.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------------------ fig 5: home vs foreign (dumbbell)
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
ax = axs[0]
for i, m in enumerate(dets):
    h = DET[(m, HOME[m])]['mAP50']
    f = DET[(m, 'eldds' if HOME[m] == 'pv' else 'pv')]['mAP50']
    ax.plot([f, h], [i, i], color=BASE, lw=2, zorder=1)
    ax.scatter([h], [i], s=70, color=COLOR[m], zorder=2, ec=SURF, lw=2)
    ax.scatter([f], [i], s=70, facecolor=SURF, ec=COLOR[m], lw=2, zorder=2)
    ax.text(h + 0.02, i, f'{h:.2f}', va='center', fontsize=8, color=INK2)
    ax.text(f, i + 0.22, f'{f:.3f}', va='top', ha='center', fontsize=8, color=INK2)
ax.set_yticks(range(len(dets)), [LAB[m] for m in dets])
ax.set_xlim(-0.05, 1)
ax.set_title('Class-agnostic mAP@.5: home (filled) vs foreign (hollow)', loc='left')
ax.grid(axis='x', color=GRID, lw=0.6)
ax.set_ylim(len(dets) - 0.35, -0.5)  # inverted, with room for the labels under the last row
ax = axs[1]
for i, m in enumerate(models):
    h = TIL[(m, HOME[m])]['auc']
    fs = [TIL[(m, d)]['auc'] for d in DATASETS if d != HOME[m]]
    ax.plot([min(fs + [h]), max(fs + [h])], [i, i], color=BASE, lw=2, zorder=1)
    ax.scatter([h], [i], s=70, color=COLOR[m], zorder=2, ec=SURF, lw=2)
    ax.scatter(fs, [i] * len(fs), s=70, facecolor=SURF, ec=COLOR[m], lw=2, zorder=2)
ax.set_yticks(range(len(models)), [LAB[m] for m in models])
ax.set_xlim(0.3, 1)
ax.axvline(0.5, color=BASE, lw=1, ls=':')
ax.set_title('Tile AUC: home (filled) vs the two foreign datasets (hollow)', loc='left')
ax.grid(axis='x', color=GRID, lw=0.6)
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(FIG / 'fig5_home_vs_foreign.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------------------ fig 6: recall per GT class
rec = {(r['model'], r['ds'], r['gt_cls']): r for r in M['recall_by_gt_class']}
fig, axs = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={'width_ratios': [5, 5], 'wspace': 0.05})
for ax, d, names, cls, lab in ((axs[0], 'pv', PV_NAMES, range(5), True), (axs[1], 'eldds', ELDDS_NAMES, range(1, 6), False)):
    mt = np.array([[rec[(m, d, c)]['recall'] if (m, d, c) in rec else np.nan for c in cls] for m in dets])
    ns = [rec[(dets[0], d, c)]['n'] if (dets[0], d, c) in rec else 0 for c in cls]
    hm = [[HOME[m] == d] * len(cls) for m in dets]
    heat(ax, mt, [LAB[m] for m in dets] if lab else [''] * len(dets), [f'{WRAP.get(names[c], names[c])}\n(n={n})' for c, n in zip(cls, ns)],
         '{:.2f}', title=f'{DS_LABEL[d]}: recall per GT class (IoU 0.5, conf 0.25)')
    if not lab:
        ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=8)
fig.savefig(FIG / 'fig6_recall_by_class.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------------------ qualitative: detections on 2 images per dataset
GT = {d: defect_gt(d, load_gt(d)) for d in ('pv', 'eldds')}
PR = {(m, d): defect_preds(m, load_preds(m, d)) for m in dets for d in DATASETS}


def pick(d, k=2):
    imgs = image_list(d)
    nd = [(len(GT[d][stem(p)]), p) for p in imgs]
    nd = [x for x in nd if 2 <= x[0] <= 12]  # readable: a handful of defects
    return [nd[i][1] for i in np.linspace(0, len(nd) - 1, k).astype(int)]


for d in ('pv', 'eldds'):
    sel = pick(d)
    fig, axs = plt.subplots(len(sel), 1 + len(dets), figsize=(3.0 * (1 + len(dets)), 3.1 * len(sel)))
    for i, p in enumerate(sel):
        im = Image.open(p).convert('L')
        s = stem(p)
        for j, m in enumerate([None] + dets):
            ax = axs[i, j]
            ax.imshow(im, cmap='gray', vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            if m is None:
                boxes, col, title = GT[d][s][:, 1:], '#e34948', 'Ground truth'
            else:
                pr = PR[(m, d)][s]
                boxes, col, title = pr[pr[:, 1] >= 0.25][:, 2:], COLOR[m], LAB[m] + (' (home)' if HOME[m] == d else '')
            for x1, y1, x2, y2 in boxes:
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=col, lw=1.6))
            if i == 0:
                ax.set_title(title, fontsize=8.5, loc='left')
            if j == 0:
                ax.set_ylabel(s[:22], fontsize=7, color=MUTED)
    fig.suptitle(f'{DS_LABEL[d]}: detections with conf >= 0.25 (all defect classes)', x=0.01, ha='left', color=INK)
    fig.tight_layout()
    fig.savefig(FIG / f'qual_det_{d}.png', dpi=110, bbox_inches='tight')
    plt.close(fig)

# ELPV cells: GT + every model's score
tiles = list(csv.DictReader(open(OUT / 'tiles.csv')))
cnn = {(t['ds'], t['image'], t['r'], t['c']): float(t['score']) for t in csv.DictReader(open(OUT / 'preds/p3_cnn__tiles.csv'))}
el = [t for t in tiles if t['ds'] == 'elpv']
choice = [t for t in el if float(t['p']) == 1.0][:4] + [t for t in el if float(t['p']) == 0][:4]
SHORT = {'p1_yolov5s': 'P1-v5s', 'p1_gbh': 'P1-GBH', 'p2_v5s': 'P2-v5s', 'p2_improved': 'P2-Impr', 'p2_v8s': 'P2-v8s'}
fig, axs = plt.subplots(2, 4, figsize=(11.5, 7.4), gridspec_kw={'hspace': 0.3, 'wspace': 0.08})
for ax, t in zip(axs.ravel(), choice):
    ax.imshow(Image.open(t['path']).convert('L'), cmap='gray', vmin=0, vmax=255)
    ax.set_xticks([])
    ax.set_yticks([])
    sc = [f'P3-CNN {cnn[("elpv", t["image"], "0", "0")]:.2f}'] + \
         [f'{SHORT[m]} {(PR[(m, "elpv")][t["image"]][:, 1].max() if len(PR[(m, "elpv")][t["image"]]) else 0):.2f}'
          for m in dets]
    ax.set_title(f'{t["image"]}  GT: {"defective" if t["label"] == "1" else "functional"} ({t["type"]})', fontsize=8.5,
                 loc='left')
    ax.set_xlabel('  '.join(sc[:3]) + '\n' + '  '.join(sc[3:]), fontsize=7.5, color=INK2)
fig.suptitle('ELPV (P3) test cells: defect score of every model (detectors: max conf of any defect box; '
             'thresholds CNN 0.5, detectors 0.25)', x=0.01, y=0.95, ha='left', color=INK)
fig.savefig(FIG / 'qual_elpv_cells.png', dpi=110, bbox_inches='tight')
plt.close(fig)

# tile view: GT tiles vs CNN vs best home detector, one ELDDS + one PV image
for d, det_m in (('eldds', 'p2_v8s'), ('pv', 'p1_yolov5s')):
    p = pick(d, 1)[0]
    s = stem(p)
    tt = [t for t in tiles if t['ds'] == d and t['image'] == s]
    im = Image.open(p).convert('L')
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, (title, val) in zip(axs, (('GT defective tiles', lambda t: float(t['label'])),
                                      ('P3 CNN tile score', lambda t: cnn[(d, s, t['r'], t['c'])]),
                                      (f'{LAB[det_m]} tile score', None))):
        ax.imshow(im, cmap='gray', vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        pr = PR[(det_m, d)][s]
        for t in tt:
            x1, y1, x2, y2 = (float(t[k]) for k in ('x1', 'y1', 'x2', 'y2'))
            if val is None:
                cx, cy = (pr[:, 2] + pr[:, 4]) / 2, (pr[:, 3] + pr[:, 5]) / 2
                ins = (cx >= x1) & (cx < x2) & (cy >= y1) & (cy < y2)
                v = pr[ins, 1].max() if ins.any() else 0.0
            else:
                v = val(t)
            v = float(np.clip(v, 0, 1))
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fc='#e34948', alpha=0.55 * v, ec='#fcfcfb', lw=0.6))
            if title != 'GT defective tiles':
                ax.text((x1 + x2) / 2, (y1 + y2) / 2, f'{v:.2f}', ha='center', va='center', fontsize=6, color='white')
        ax.set_title(title, loc='left', fontsize=9)
    fig.suptitle(f'{DS_LABEL[d]} {s[:30]}: tile task (red intensity = score)', x=0.01, ha='left', color=INK)
    fig.tight_layout()
    fig.savefig(FIG / f'qual_tiles_{d}.png', dpi=110, bbox_inches='tight')
    plt.close(fig)

# ------------------------------------------------------------------ tables
L = []
L.append('### Detection, class-agnostic (defect vs background)\n')
L.append('| Model | Dataset | home? | P | R | mAP@.5 [95% CI] | mAP@.5:.95 |')
L.append('|---|---|---|---|---|---|---|')
for m in dets:
    for d in ('pv', 'eldds'):
        r = DET[(m, d)]
        L.append(f'| {LAB[m]} | {DS_LABEL[d]} | {"home" if HOME[m] == d else "cross"} | {r["P"]:.3f} | {r["R"]:.3f} | '
                 f'{r["mAP50"]:.3f} [{r["mAP50_ci"][0]:.3f}, {r["mAP50_ci"][1]:.3f}] | {r["mAP"]:.3f} |')
L.append('\n### Detection, class-aware on the home dataset (our evaluator; paper metric)\n')
L.append('| Model | Dataset | P | R | mAP@.5 | mAP@.5 defect classes only | mAP@.5:.95 | AP@.5 per class |')
L.append('|---|---|---|---|---|---|---|---|')
for m in dets:
    r = DET_HOME[m]
    names = PV_NAMES if HOME[m] == 'pv' else ELDDS_NAMES
    pc = ', '.join(f'{names[int(c)]} {v:.3f}' for c, v in r['ap50_per_class'].items())
    dm = np.mean([v for c, v in r['ap50_per_class'].items() if not (HOME[m] == 'eldds' and int(c) == 0)])
    L.append(f'| {LAB[m]} | {DS_LABEL[HOME[m]]} | {r["P"]:.3f} | {r["R"]:.3f} | {r["mAP50"]:.3f} | {dm:.3f} | '
             f'{r["mAP"]:.3f} | {pc} |')
L.append('\n### Tile classification (defective vs functional)\n')
L.append('| Model | Dataset | home? | n tiles (defective) | AUC [95% CI] | Acc | macro-F1 [95% CI] | P_def | R_def | F1_def |')
L.append('|---|---|---|---|---|---|---|---|---|---|')
for m in models:
    for d in DATASETS:
        r = TIL[(m, d)]
        L.append(f'| {LAB[m]} | {DS_LABEL[d]} | {"home" if HOME[m] == d else "cross"} | {r["n"]} ({r["n_defective"]}) | '
                 f'{r["auc"]:.3f} [{r["auc_ci"][0]:.3f}, {r["auc_ci"][1]:.3f}] | {r["acc"]:.3f} | '
                 f'{r["f1_macro"]:.3f} [{r["f1_macro_ci"][0]:.3f}, {r["f1_macro_ci"][1]:.3f}] | {r["precision"]:.3f} | '
                 f'{r["recall"]:.3f} | {r["f1_defective"]:.3f} |')
L.append('\n### ELPV with P3\'s confidence weights (P3 protocol)\n')
L.append('| Model | acc_w | macro-F1_w | AUC_w |')
L.append('|---|---|---|---|')
for m in models:
    r = TIL[(m, 'elpv')]
    L.append(f'| {LAB[m]} | {100 * r["acc_w"]:.2f} | {100 * r["f1_macro_w"]:.2f} | {100 * r["auc_w"]:.2f} |')
L.append('\n### Inference speed on this machine (RTX 4050 Laptop, batch 1 for detectors, fp16)\n')
L.append('| Model | input px | ms/img PV | ms/img ELDDS | ms/img ELPV | NMS ms/img (ELDDS) |')
L.append('|---|---|---|---|---|---|')
for m in dets:
    sp = [M['speed'][f'{m}__{d}'] for d in DATASETS]
    L.append(f'| {LAB[m]} | {sp[0]["img"]} | {sp[0]["ms_inference"]:.1f} | {sp[1]["ms_inference"]:.1f} | '
             f'{sp[2]["ms_inference"]:.1f} | {sp[1]["ms_nms"]:.1f} |')
c = M['speed']['p3_cnn__tiles']
L.append(f'| {LAB["p3_cnn"]} | 300 | {c["ms_per_tile_batch16"]:.1f} per tile (batch 16, fp32) | | | |')
(OUT / 'tables.md').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
