"""Frozen 75/25 train/test split of the ELPV dataset for P3 (Deitsch et al. 2019).

The paper split 2624 cells into 1968 train / 656 test by stratified random sampling but did not publish the split.
We reproduce its Table 3 exactly: per module type (mono/poly) and label p (0, 1/3, 2/3, 1) the same train/test counts.
Membership within each stratum is seeded random (seed 0). Existing split files are reused, never regenerated.

usage: python -I elpv_split.py --data datasets/raw/elpv-dataset/src/elpv_dataset/data --out datasets/splits/elpv
"""
import argparse
import random
from collections import Counter
from pathlib import Path

# Table 3 of the paper: test counts per (type, p); train = total - test
TEST = {('mono', 0.0): 150, ('mono', 1 / 3): 30, ('mono', 2 / 3): 15, ('mono', 1.0): 64,
        ('poly', 0.0): 237, ('poly', 1 / 3): 46, ('poly', 2 / 3): 13, ('poly', 1.0): 101}
TRAIN = {('mono', 0.0): 438, ('mono', 1 / 3): 87, ('mono', 2 / 3): 41, ('mono', 1.0): 249,
         ('poly', 0.0): 683, ('poly', 1 / 3): 132, ('poly', 2 / 3): 37, ('poly', 1.0): 301}


def load(data):
    rows = [l.split() for l in (Path(data) / 'labels.csv').read_text().splitlines() if l.strip()]
    return [(path, round(float(p), 4), t) for path, p, t in rows]


def key(p, t):
    return t, min((0.0, 1 / 3, 2 / 3, 1.0), key=lambda v: abs(v - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    rows = load(a.data)
    assert len(rows) == 2624, len(rows)
    if (out / 'train.txt').exists() and (out / 'test.txt').exists():
        print(f'reusing frozen split in {out}')
    else:
        rng = random.Random(a.seed)
        train, test = [], []
        for k in sorted(TEST, key=str):
            s = sorted(r[0] for r in rows if key(r[1], r[2]) == k)
            assert len(s) == TEST[k] + TRAIN[k], (k, len(s))
            rng.shuffle(s)
            test += s[:TEST[k]]
            train += s[TEST[k]:]
        out.mkdir(parents=True, exist_ok=True)
        (out / 'train.txt').write_text(''.join(f'{p}\n' for p in sorted(train)))
        (out / 'test.txt').write_text(''.join(f'{p}\n' for p in sorted(test)))
        print(f'wrote new split to {out} (seed {a.seed})')
    lab = {r[0]: r for r in rows}
    for name, want in (('train', TRAIN), ('test', TEST)):
        paths = (out / f'{name}.txt').read_text().split()
        c = Counter(key(lab[p][1], lab[p][2]) for p in paths)
        assert dict(c) == want, (name, c)
        print(f'{name}: {len(paths)} cells, Table 3 counts match')


if __name__ == '__main__':
    main()
