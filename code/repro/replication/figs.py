"""Report figures that combine both replications (R1 = earlier RTX 3050 run, R2 = main run).

fig9:  each paper's reported home score vs R1 and R2 (P1 mAP@.5, P2 mAP@.5, P3 confidence-weighted accuracy).
fig10: foreign-dataset tile AUC of every P1 model of both runs, with 95 % CIs (results/replication_r1/summary.json).

usage (repo root): envs/p3cnn/Scripts/python.exe -I code/repro/replication/figs.py  -> results/cross/figs/fig9_*, fig10_*
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
FIG = ROOT / 'results/cross/figs'
INK, INK2, MUTED, GRID, BASE, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
R1C, R2C = '#2a78d6', '#eb6834'  # categorical slots 1-2, validated (light surface)
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'axes.edgecolor': BASE, 'axes.labelcolor': INK2,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.titlecolor': INK, 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False, 'axes.titlesize': 11})
S = json.load(open(ROOT / 'results/replication_r1/summary.json'))
r1 = {m: 100 * S['p1'][m]['r1_home']['mAP50'] for m in ('r1_yolov5s', 'r1_yolov5_1', 'r1_yolov5_2', 'r1_gbh')}

# (model, paper, R1, R2); None = not available. P1/P2: official val scripts; P3: our evaluator, acc_w
PANELS = [
    ('P1 · PV-Multi-Defect · mAP@.5 (%)', [
        ('YOLOv5s', 78.1, r1['r1_yolov5s'], 86.5),
        ('YOLOv5-1 (+BottleneckCSP)', 94.2, r1['r1_yolov5_1'], 85.2),
        ('YOLOv5-2 (+tiny-target head)', 97.1, r1['r1_yolov5_2'], None),
        ('GBH-YOLOv5', 97.8, r1['r1_gbh'], 82.1)]),
    ('P2 · ELDDS1400C5 · mAP@.5 (×100)', [
        ('YOLOv5s', 73.9, None, 75.4),
        ('YOLOv5s+GAM+ASFF+DIoU', 76.2, None, 75.7),
        ('YOLOv8s', 76.5, None, 76.4)]),
    ('P3 · ELPV · weighted accuracy (%)', [
        ('VGG-19 CNN', 88.42, 100 * S['p3']['cnn']['acc_w'], 86.49),
        ('SVM KAZE/VGG, linear', 82.44, 100 * S['p3']['svm_linear']['acc_w'], 79.65)]),
]
fig, axs = plt.subplots(3, 1, figsize=(8.2, 6.4), gridspec_kw={'height_ratios': [4, 3, 2]})
for ax, (title, rows) in zip(axs, PANELS):
    for i, (name, paper, a, b) in enumerate(rows):
        y = len(rows) - 1 - i
        vals = [v for v in (paper, a, b) if v is not None]
        ax.plot([min(vals), max(vals)], [y, y], color=GRID, lw=2, zorder=1, solid_capstyle='round')
        ax.scatter(paper, y, marker='D', s=46, facecolor=SURF, edgecolor=INK, lw=1.6, zorder=3)
        if a is not None:
            ax.scatter(a, y, marker='o', s=58, color=R1C, edgecolor=SURF, lw=2, zorder=4)
        if b is not None:
            ax.scatter(b, y, marker='s', s=52, color=R2C, edgecolor=SURF, lw=2, zorder=4)
        ax.text(paper, y + 0.3, f'{paper:.1f}', ha='center', va='bottom', fontsize=8, color=INK2)
        best = max(v for v in (a, b) if v is not None)
        ax.text(best, y - 0.3, f'{best:.1f}', ha='center', va='top', fontsize=8, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], color=INK2)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_title(title, loc='left')
    ax.grid(axis='x', color=GRID, lw=0.6)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
axs[0].set_xlim(76, 100)
axs[1].set_xlim(72, 78)
axs[2].set_xlim(76, 90)
handles = [Line2D([], [], marker='D', ls='', mfc=SURF, mec=INK, mew=1.6, ms=7, label='Paper reports'),
           Line2D([], [], marker='o', ls='', color=R1C, ms=8, label='R1 (earlier run, 600 px)'),
           Line2D([], [], marker='s', ls='', color=R2C, ms=7.5, label='R2 (main run)')]
fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.03), fontsize=9, labelcolor=INK2)
fig.text(0.01, -0.07, 'Each run on its own held-out split. Value below a row = best reproduction. P2 has one run (R2); '
         'R1 did not reproduce P2. P3 R1 re-scored with our evaluator.', color=MUTED, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / 'fig9_paper_vs_reproduction.png', dpi=150, bbox_inches='tight')

# ---------------------------------------------------------------- fig10
ORDER = [('YOLOv5s', 'r1_yolov5s', 'r2_yolov5s'), ('YOLOv5-1', 'r1_yolov5_1', 'r2_yolov5_1'),
         ('YOLOv5-2', 'r1_yolov5_2', None), ('GBH-YOLOv5', 'r1_gbh', 'r2_gbh')]
fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.0), sharey=True)
for ax, ds, title in zip(axs, ('eldds', 'elpv'), ('P1 models on ELDDS1400C5 (P2 data)', 'P1 models on ELPV (P3 data)')):
    ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(0.497, -0.45, 'chance', color=MUTED, fontsize=8, va='center', ha='right')
    for i, (name, a, b) in enumerate(ORDER):
        y = len(ORDER) - 1 - i
        for m, off, col, mk in ((a, 0.14, R1C, 'o'), (b, -0.14, R2C, 's')):
            if m is None:
                continue
            r = S['p1'][m]
            lo, hi = r[f'{ds}_tile_auc_ci']
            ax.plot([lo, hi], [y + off] * 2, color=col, lw=2, solid_capstyle='round', zorder=2)
            ax.scatter(r[f'{ds}_tile_auc'], y + off, marker=mk, s=40, color=col, edgecolor=SURF, lw=1.5, zorder=3)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([o[0] for o in ORDER][::-1], color=INK2)
    ax.set_xlim(0.44, 0.68)
    ax.set_title(title, loc='left', fontsize=10)
    ax.set_xlabel('defective-vs-functional tile AUC (95 % CI)')
    ax.grid(axis='x', color=GRID, lw=0.6)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
handles = [Line2D([], [], marker='o', color=R1C, ms=7, lw=2, label='R1 (600 px)'),
           Line2D([], [], marker='s', color=R2C, ms=6.5, lw=2, label='R2 (960 px)')]
fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.1), fontsize=9, labelcolor=INK2)
fig.text(0.01, -0.17, 'Home tile AUC of these models: 0.98 (PV-Multi-Defect). R2 YOLOv5-2 was not trained to completion.',
         color=MUTED, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / 'fig10_replication_foreign_auc.png', dpi=150, bbox_inches='tight')
sys.stdout.write('wrote fig9, fig10\n')
