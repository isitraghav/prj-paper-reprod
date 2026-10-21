"""Write results/cross/tiles.csv (tile grid + labels, see common.py) and check the assumed image sizes.

usage (repo root): envs/p3cnn/Scripts/python.exe -I code/repro/cross/make_tiles.py [--eldds-grid 12 --out tiles_eldds12.csv]
"""
import argparse
import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from PIL import Image  # noqa: E402

from common import DATASETS, GRID, OUT, image_list, image_size, make_tiles  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--eldds-grid', type=int, default=GRID['eldds'])
ap.add_argument('--out', default='tiles.csv')
a = ap.parse_args()

for ds in DATASETS:
    sizes = collections.Counter(Image.open(p).size for p in image_list(ds))
    assert set(sizes) == {image_size(ds)}, (ds, sizes)
tiles = make_tiles({**GRID, 'eldds': a.eldds_grid})
with open(OUT / a.out, 'w', newline='') as f:
    w = csv.DictWriter(f, list(tiles[0]))
    w.writeheader()
    w.writerows(tiles)
for ds in DATASETS:
    t = [x for x in tiles if x['ds'] == ds]
    print(ds, 'tiles', len(t), 'defective', sum(x['label'] for x in t), 'images', len({x['image'] for x in t}))
