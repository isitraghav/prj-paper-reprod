"""P3 CNN (Deitsch et al. 2019, Sec. 3.2): ImageNet VGG-19 fine-tuned as a deep regression network on ELPV.

Paper -> here:
- VGG-19 conv blocks (ImageNet, Keras weights); top replaced by GAP + FC 4096 + FC 2048 + 1 output neuron (defect
  probability), MSE loss, regression targets p in {0, 1/3, 2/3, 1}. Input 300x300x3 (grayscale replicated),
  Keras VGG preprocessing. Hidden FC: ReLU; output: linear (regression; activation not stated in the paper --
  a sigmoid output saturated and did not train in a smoke test, see init.md).
- Augmentation: scale <= 2%, rotation <= 3 deg, translation <= 2%, random horizontal + vertical flips, plus samples
  rotated by exactly 90 deg (here: with probability 0.5, so an epoch stays 1968 samples = 196800 over 100 epochs).
- Stage 1: only the new FC layers (conv frozen), Adam lr 1e-3, beta1 0.9, beta2 0.999, eps 1e-8.
  Stage 2: all layers, SGD lr 5e-4, momentum 0.9. Both: mini-batches of 16, 100 epochs (full, final weights).
- Class balancing (paper Eq. 5) as per-sample weights c_j.
- Batch 16 is exact: if it does not fit, gradients of --micro sized chunks are accumulated; VGG-19 has no BatchNorm,
  so this is mathematically the same update as one batch of 16.

usage (repo root): envs/p3cnn/Scripts/python.exe -I code/repro/p3/train_cnn.py --out results/p3/cnn [--epochs 100]
       TensorFlow 2.12 (Docker, GPU): bash code/repro/p3/cnn_tf212.sh <seed>. Training needs a GPU; without one it exits.
"""
import argparse
import csv
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuda_path  # noqa: E402,F401  (must precede tensorflow)
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from PIL import Image  # noqa: E402

from elpv_common import class_weights, fmt_metrics, load_split, metrics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='.')
ap.add_argument('--out', required=True)
ap.add_argument('--epochs', type=int, default=100)
ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--micro', type=int, default=0, help='stage-2 micro-batch for accumulation (0 = auto)')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--stages', default='12')
ap.add_argument('--out-act', default='linear', choices=['linear', 'sigmoid'])
ap.add_argument('--no-eval', action='store_true')
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
random.seed(a.seed)
np.random.seed(a.seed)
tf.random.set_seed(a.seed)
if not tf.config.list_physical_devices('GPU'):  # GPU only: never fall back to training on the CPU
    sys.exit('TensorFlow sees no GPU: refusing to train on the CPU')
for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)
LOG = open(out / 'train.log', 'a', buffering=1)


def log(*m):
    s = time.strftime('[%F %T] ') + ' '.join(map(str, m))
    print(s, flush=True)
    LOG.write(s + '\n')


log('tf', tf.__version__, 'gpus', tf.config.list_physical_devices('GPU'), 'args', vars(a))

# ------------------------------------------------------------------ data
def load_images(paths):
    return np.stack([np.asarray(Image.open(p).convert('L'), dtype=np.uint8) for p in paths])


tr_paths, tr_p, tr_t = load_split(a.root, 'train')
te_paths, te_p, te_t = load_split(a.root, 'test')
Xtr, Xte = load_images(tr_paths), load_images(te_paths)
cw = class_weights(tr_p)
sw = np.where(tr_p > 0, cw[1], cw[0]).astype(np.float32)
log(f'train {Xtr.shape} test {Xte.shape} class weights functional {cw[0]:.4f} defective {cw[1]:.4f}')
assert Xtr.shape == (1968, 300, 300) and Xte.shape == (656, 300, 300)

gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=3, width_shift_range=0.02, height_shift_range=0.02, zoom_range=0.02,
    horizontal_flip=True, vertical_flip=True, fill_mode='nearest')
rng = np.random.RandomState(a.seed)
preprocess = tf.keras.applications.vgg19.preprocess_input


def to_input(gray):  # (n,300,300) uint8/float -> VGG input (n,300,300,3) float32
    x = np.repeat(np.asarray(gray, np.float32)[..., None], 3, axis=-1)
    return preprocess(x)


def augment(img):  # (300,300) uint8
    x = img[..., None].astype(np.float32)
    if rng.rand() < 0.5:
        x = np.rot90(x, 1, axes=(0, 1))
    return gen.random_transform(x, seed=rng.randint(2 ** 31 - 1))[..., 0]


def epoch_batches():
    idx = rng.permutation(len(Xtr))
    for s in range(0, len(idx), a.batch):
        b = idx[s:s + a.batch]
        yield to_input(np.stack([augment(Xtr[i]) for i in b])), tr_p[b].astype(np.float32), sw[b]


# ------------------------------------------------------------------ model
base = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(300, 300, 3))
x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(base.output)
x = tf.keras.layers.Dense(4096, activation='relu', name='fc4096')(x)
x = tf.keras.layers.Dense(2048, activation='relu', name='fc2048')(x)
y = tf.keras.layers.Dense(1, activation=None if a.out_act == 'linear' else 'sigmoid', name='defect_probability')(x)
model = tf.keras.Model(base.input, y)
log('params', model.count_params())


def make_step(opt, micro):
    @tf.function
    def grads_of(xb, yb, wb):
        with tf.GradientTape() as tape:
            pred = model(xb, training=True)[:, 0]
            loss = tf.reduce_sum(wb * tf.square(pred - yb)) / tf.cast(a.batch, tf.float32)  # Keras SUM_OVER_BATCH_SIZE
        return loss, tape.gradient(loss, model.trainable_variables)

    def step(xb, yb, wb):
        tot, acc = 0.0, None
        for s in range(0, len(xb), micro):
            l, g = grads_of(xb[s:s + micro], yb[s:s + micro], wb[s:s + micro])
            tot += float(l)
            acc = g if acc is None else [u + v for u, v in zip(acc, g)]
        opt.apply_gradients(zip(acc, model.trainable_variables))
        return tot
    return step


def fits(micro):
    try:
        xb = to_input(Xtr[:micro])
        with tf.GradientTape() as tape:
            l = tf.reduce_mean(model(xb, training=True))
        tape.gradient(l, model.trainable_variables)
        return True
    except tf.errors.ResourceExhaustedError:
        return False


def predict(X):
    return np.concatenate([model(to_input(X[s:s + 16]), training=False).numpy()[:, 0] for s in range(0, len(X), 16)])


def run_stage(stage, opt, epochs):
    micro = a.batch
    if stage == 2:
        micro = a.micro or next((m for m in (16, 8, 4, 2) if fits(m)), 1)
    log(f'stage {stage}: trainable vars {len(model.trainable_variables)}, batch {a.batch} as {a.batch // micro} x {micro}')
    step = make_step(opt, micro)
    # per-epoch checkpoint (weights + optimizer slots/iterations + augmentation RNG) -> exact resume after a kill
    ckpt = tf.train.Checkpoint(model=model, optimizer=opt)
    cdir, sfile = out / f'ckpt_stage{stage}', out / f'ckpt_stage{stage}.json'
    start = 1
    if sfile.exists():
        st = json.loads(sfile.read_text())
        ckpt.restore(st['path'])
        rng.set_state(pickle.loads(bytes.fromhex(st['rng'])))
        start = st['epoch'] + 1
        log(f'stage {stage}: resumed after epoch {st["epoch"]} from {st["path"]}, optimizer iterations {int(opt.iterations)}')
    with open(out / f'stage{stage}_history.csv', 'a', newline='') as f:
        w = csv.writer(f)
        for ep in range(start, epochs + 1):
            t0, losses = time.time(), []
            for xb, yb, wb in epoch_batches():
                losses.append(step(xb, yb, wb))
            w.writerow([ep, np.mean(losses), time.time() - t0])
            f.flush()
            path = ckpt.write(str(cdir / 'ckpt'))
            sfile.write_text(json.dumps({'epoch': ep, 'path': path, 'rng': pickle.dumps(rng.get_state()).hex()}))
            log(f'stage {stage} epoch {ep}/{epochs} loss {np.mean(losses):.5f} ({time.time() - t0:.0f}s)')
    model.save_weights(str(out / f'stage{stage}.h5'))


if '1' in a.stages and not (out / 'stage1.h5').exists():
    base.trainable = False
    run_stage(1, tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8), a.epochs)
elif (out / 'stage1.h5').exists():
    model.load_weights(str(out / 'stage1.h5'))
    if (out / 'ckpt_stage1.json').exists() and not (out / 'ckpt_stage2.json').exists():  # RNG as after stage 1
        rng.set_state(pickle.loads(bytes.fromhex(json.loads((out / 'ckpt_stage1.json').read_text())['rng'])))
    log('stage 1 already complete: loaded stage1.h5')
if '2' in a.stages:
    base.trainable = True
    run_stage(2, tf.keras.optimizers.SGD(learning_rate=5e-4, momentum=0.9), a.epochs)

# ------------------------------------------------------------------ evaluate once on the test split (final weights)
if a.no_eval:
    sys.exit(0)
score = predict(Xte)
m = metrics(te_p, score, te_t, threshold=0.5)
log('TEST (final weights)\n' + fmt_metrics(m))
json.dump(m, open(out / 'test_metrics.json', 'w'), indent=1)
with open(out / 'test_predictions.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['path', 'p_true', 'type', 'p_pred'])
    w.writerows(zip([os.path.basename(p) for p in te_paths], te_p, te_t, score))
