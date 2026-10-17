"""Evaluate a saved P3 CNN (train_cnn.py weights .h5) on the ELPV test split: threshold 0.5, same metrics as training.

usage (repo root): envs/p3cnn/Scripts/python.exe -I code/repro/p3/eval_cnn.py models/p3/cnn_vgg19_seed0.h5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuda_path  # noqa: E402,F401  (must precede tensorflow)
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from PIL import Image  # noqa: E402

from elpv_common import fmt_metrics, load_split, metrics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('weights')
ap.add_argument('--root', default='.')
a = ap.parse_args()
for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)

base = tf.keras.applications.VGG19(include_top=False, weights=None, input_shape=(300, 300, 3))
x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(base.output)
x = tf.keras.layers.Dense(4096, activation='relu', name='fc4096')(x)
x = tf.keras.layers.Dense(2048, activation='relu', name='fc2048')(x)
model = tf.keras.Model(base.input, tf.keras.layers.Dense(1, name='defect_probability')(x))
model.load_weights(a.weights)

paths, p, t = load_split(a.root, 'test')
X = np.stack([np.asarray(Image.open(q).convert('L'), dtype=np.float32) for q in paths])
pre = tf.keras.applications.vgg19.preprocess_input
score = np.concatenate([model(pre(np.repeat(X[s:s + 16, ..., None], 3, -1)), training=False).numpy()[:, 0]
                        for s in range(0, len(X), 16)])
print(fmt_metrics(metrics(p, score, t, threshold=0.5)))
