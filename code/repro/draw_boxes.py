"""Draw boxes for N random images of a YOLO-format list file, for an eyeball check.

usage: python -I draw_boxes.py --data datasets/pv_multi_defect_yolo/data.yaml --split train \
           --n 5 --seed 0 --out results/sanity/p1
"""
import argparse
import random
from pathlib import Path

import cv2
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--n', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    d = yaml.safe_load(open(a.data))
    names = d['names']
    ims = Path(d[a.split]).read_text().split()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    colors = [(rng.randint(60, 255), rng.randint(60, 255), rng.randint(60, 255)) for _ in names]
    for p in rng.sample(ims, a.n):
        im = cv2.imread(p)
        h, w = im.shape[:2]
        lab = Path(p.replace('/images/', '/labels/')).with_suffix('.txt')
        for l in lab.read_text().splitlines():
            c, x, y, bw, bh = l.split()
            c, x, y, bw, bh = int(c), float(x) * w, float(y) * h, float(bw) * w, float(bh) * h
            p1, p2 = (int(x - bw / 2), int(y - bh / 2)), (int(x + bw / 2), int(y + bh / 2))
            cv2.rectangle(im, p1, p2, colors[c], 2)
            cv2.putText(im, names[c], (p1[0], max(p1[1] - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[c], 1)
        cv2.imwrite(str(out / Path(p).name), im)
        print(out / Path(p).name)


if __name__ == '__main__':
    main()
