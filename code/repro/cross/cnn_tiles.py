"""P3 CNN (models/p3/cnn_vgg19_seed0.h5, unchanged) on every tile of results/cross/tiles.csv.

Exactly the P3 inference path (eval_cnn.py): grayscale crop -> 300x300 (bilinear; ELPV cells are already 300x300),
replicated to 3 channels, vgg19.preprocess_input, linear output = defect probability. Writes results/cross/preds/
p3_cnn__tiles.csv (tile key + score) and the mean inference time per tile.

usage (repo root): envs/p3cnn/Scripts/python.exe -I code/repro/cross/cnn_tiles.py [--tiles tiles.csv --out p3_cnn__tiles]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'p3')]
import cuda_path  # noqa: E402,F401  (must precede tensorflow)
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from PIL import Image  # noqa: E402

from common import OUT, ROOT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--tiles', default='tiles.csv')
ap.add_argument('--out', default='p3_cnn__tiles')
a = ap.parse_args()

for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)
base = tf.keras.applications.VGG19(include_top=False, weights=None, input_shape=(300, 300, 3))
x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(base.output)
x = tf.keras.layers.Dense(4096, activation='relu', name='fc4096')(x)
x = tf.keras.layers.Dense(2048, activation='relu', name='fc2048')(x)
model = tf.keras.Model(base.input, tf.keras.layers.Dense(1, name='defect_probability')(x))
model.load_weights(str(ROOT / 'models/p3/cnn_vgg19_seed0.h5'))
pre = tf.keras.applications.vgg19.preprocess_input

tiles = list(csv.DictReader(open(OUT / a.tiles)))
cache, crops = {}, []
for t in tiles:
    if t['path'] not in cache:
        cache = {t['path']: Image.open(t['path']).convert('L')}  # tiles are grouped by image
    im = cache[t['path']]
    box = tuple(int(round(float(t[k]))) for k in ('x1', 'y1', 'x2', 'y2'))
    crop = im.crop(box)
    if crop.size != (300, 300):
        crop = crop.resize((300, 300), Image.BILINEAR)
    crops.append(np.asarray(crop, np.uint8))  # uint8: ~11k tiles x 90 KB
X = np.stack(crops)
print('tiles', X.shape, flush=True)

to_input = lambda b: pre(np.repeat(b.astype(np.float32)[..., None], 3, -1))
model(to_input(X[:16]), training=False)  # warm-up
t0, scores = time.time(), []
for s in range(0, len(X), 16):
    scores.append(model(to_input(X[s:s + 16]), training=False).numpy()[:, 0])
dt = time.time() - t0
score = np.concatenate(scores)
with open(OUT / 'preds' / f'{a.out}.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['ds', 'image', 'r', 'c', 'score'])
    w.writerows([t['ds'], t['image'], t['r'], t['c'], f'{v:.5f}'] for t, v in zip(tiles, score))
meta = {'n_tiles': len(X), 'ms_per_tile_batch16': 1000 * dt / len(X), 'tf': tf.__version__,
        'gpu': str(tf.config.list_physical_devices('GPU'))}
json.dump(meta, open(OUT / 'preds' / f'{a.out}.json', 'w'), indent=1)
print(meta)
