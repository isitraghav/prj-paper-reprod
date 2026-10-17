"""Collect P2 evaluation logs into results/summary.csv and results/summary.md (paper vs ours vs delta).

usage: python -I code/repro/summarize.py            (from the repo root)
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / 'results' / 'p2' / 'eval'
MACHINE = 'Laptop B (RTX 3050 Laptop 6 GB, Win 11)'
BATCH = '16 x 4 accum (paper 32 x 2)'

# P2 Table 1 (test set, no TTA unless stated) and Table 2 (mAP@.5 w/o / with TTA)
PAPER = {
    'v5s': dict(model='YOLOv5s_only', P=0.726, R=0.723, map50=0.738, map=0.502, map50_t2=0.739, map50_tta=0.766),
    'v5_improved': dict(model='YOLOv5s+GAM+ASFF+DIoU', P=0.735, R=0.760, map50=0.763, map=0.516, map50_t2=0.76222,
                        map50_tta=0.777),
    'v8s': dict(model='YOLOv8s', P=None, R=None, map50=None, map=None, map50_t2=0.765, map50_tta=0.775),
}
ALL = re.compile(r'^\s+all\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)')


def parse(log):
    if not log.exists():
        return None
    lines = re.sub(r'\x1b\[[0-9;]*m', '', log.read_text(encoding='utf8', errors='replace').replace('\r', '\n')).splitlines()
    rows = [m for m in map(ALL.match, lines) if m]
    if not rows:
        return None
    m = rows[-1]
    return dict(P=float(m[3]), R=float(m[4]), map50=float(m[5]), map=float(m[6]))


def fmt(x, nd=3):
    return '—' if x is None else f'{x:.{nd}f}'


def delta(a, b):
    return '—' if a is None or b is None else f'{a - b:+.3f}'


rows = []
for run, p in PAPER.items():
    n, t = parse(EVAL / f'{run}_notta.log'), parse(EVAL / f'{run}_tta.log')
    rows.append(dict(run=run, model=p['model'],
                     P=n and n['P'], R=n and n['R'], map50=n and n['map50'], map=n and n['map'],
                     map50_tta=t and t['map50'], paper_P=p['P'], paper_R=p['R'], paper_map50=p['map50'],
                     paper_map=p['map'], paper_map50_t2=p['map50_t2'], paper_map50_tta=p['map50_tta'],
                     machine=MACHINE, batch=BATCH))

out = ROOT / 'results'
with open(out / 'summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

md = ['# P2 reproduction summary (ELDDS1400C5 test split, 190 images, 640 px)', '',
      f'Machine: {MACHINE}. Batch: {BATCH}. Settings and deviations: `init.md` §3, §11.', '',
      '| Model | P (paper / ours) | R (paper / ours) | mAP@.5 (paper T1 / ours / Δ) | mAP@.5:.95 (paper / ours / Δ) '
      '| mAP@.5 no TTA (paper T2 / ours / Δ) | mAP@.5 TTA (paper T2 / ours / Δ) |',
      '|---|---|---|---|---|---|---|']
for r in rows:
    md.append(f"| {r['model']} | {fmt(r['paper_P'])} / {fmt(r['P'])} | {fmt(r['paper_R'])} / {fmt(r['R'])} "
              f"| {fmt(r['paper_map50'])} / {fmt(r['map50'])} / {delta(r['map50'], r['paper_map50'])} "
              f"| {fmt(r['paper_map'])} / {fmt(r['map'])} / {delta(r['map'], r['paper_map'])} "
              f"| {fmt(r['paper_map50_t2'])} / {fmt(r['map50'])} / {delta(r['map50'], r['paper_map50_t2'])} "
              f"| {fmt(r['paper_map50_tta'])} / {fmt(r['map50_tta'])} / {delta(r['map50_tta'], r['paper_map50_tta'])} |")
(out / 'summary.md').write_text('\n'.join(md) + '\n', encoding='utf8')
print('\n'.join(md))
