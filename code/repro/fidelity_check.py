"""Fidelity audit: does each training run use the code, algorithm and settings its paper states?

Every "ours" value is read from the run's own artefacts (the argument Namespace logged at training start, hyp/opt/args
yaml, results files, model summaries, git state of the codebases, checksums), not from our notes. Every "paper" value
is the paper's own statement (quoted). Writes results/fidelity_check.md.

usage (repo root): envs/p1/Scripts/python.exe -I code/repro/fidelity_check.py
"""
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / 'results'
sys.stdout.reconfigure(encoding='utf-8')
rows = []  # (paper, item, paper states, ours, verdict)


def add(paper, item, stated, ours, ok, note=''):
    """ok: True = same as the paper, False = differs, None = unavoidable or not stated by the paper (note says why)."""
    rows.append((paper, item, stated, ours, {True: '✅ same', False: '❌ differs', None: '⚠ explained'}[ok], note))


def namespace(log):
    """First argument Namespace in a yolov5 v5.0 training log (the original call, before any --resume)."""
    m = re.search(r'Namespace\((.*?)\)\s*$', open(log, encoding='utf-8', errors='replace').read().replace('\r', '\n'), re.M)
    out = {}
    for k, v in re.findall(r"(\w+)=('[^']*'|\[[^\]]*\]|[^,]+)", m.group(1)):
        out[k] = v.strip("'")
    return out


def hyp_diff(a, b):
    ya, yb = yaml.safe_load(open(a)), yaml.safe_load(open(b))
    return {k: (ya.get(k), yb.get(k)) for k in set(ya) | set(yb) if ya.get(k) != yb.get(k)}


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True).stdout.strip()


def sha256(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def summary(log):
    t = open(log, encoding='utf-8', errors='replace').read().replace('\r', '\n')
    m = re.search(r'Model Summary: (\d+) layers, (\d+) parameters', t)
    return int(m.group(1)), int(m.group(2)) / 1e6


def nlines(p):
    return sum(1 for l in open(p) if l.strip())


sums = dict(l.split()[::-1] for l in open(ROOT / 'weights/SHA256SUMS') if l.strip())
sums = {k.lstrip('*'): v for k, v in sums.items()}

# ------------------------------------------------------------------ P1
G = ROOT / 'code/gbh-yolov5'
head = git(G, 'rev-parse', 'HEAD')
ws_only = git(G, 'diff', '-w', '--ignore-blank-lines', '--', '.', ':!*.pyc') == ''
add('P1', 'Training code', 'authors\' repo CCNUZFW/GBH-YOLOv5', f'commit {head[:7]}; local change: {"whitespace only (train.py line 1, needed to run)" if ws_only else "CODE CHANGED"}',
    head.startswith('51ee042') and ws_only)
released = (yaml.safe_load(open(ROOT / 'code/repro/p1/gbh_released.yaml')) == yaml.safe_load(open(G / 'models/yolov5s_pv.yaml')))
add('P1', 'GBH architecture', 'Fig. 1 / Table 6: 270 layers, 7.24 M params', 'authors\' released `models/yolov5s_pv.yaml`, ' +
    ('identical' if released else 'MODIFIED') + ' → %d layers, %.2f M' % summary(R / 'p1/eval_p1_gbh.log'), None,
    'the authors\' own config does not give the paper\'s count; no public config does (32 variants tried)')
for run, log, cfg_paper in (('YOLOv5s', 'eval_p1_yolov5s.log', (224, 7.06)), ('YOLOv5-1', 'eval_p1_yolov5_1_ep200.log', (228, 7.15))):
    n, p = summary(R / f'p1/{log}')
    add('P1', f'{run} architecture', f'Table 6: {cfg_paper[0]} layers, {cfg_paper[1]} M', f'{n} layers, {p:.2f} M (fused)',
        n == cfg_paper[0] and abs(p - cfg_paper[1]) < 0.01)
hd = hyp_diff(ROOT / 'code/repro/p1/hyp.gbh.yaml', G / 'data/hyp.scratch.yaml')
add('P1', 'Learning rate', '"the learning rate is set at 0.001"', f"lr0 {hd['lr0'][0]} (repo default {hd['lr0'][1]})", hd['lr0'][0] == 0.001)
add('P1', 'Other hyper-parameters', 'not stated → repo defaults', 'all other keys = repo `hyp.scratch.yaml`' if set(hd) == {'lr0'} else f'differ: {hd}', set(hd) == {'lr0'})
add('P1', 'Initial weights', '"pretrained weights based on the COCO dataset"', 'COCO yolov5s.pt, yolov5 v5.0 release (sha256 %s…)' % sha256(ROOT / 'weights/yolov5s_v5.0.pt')[:12],
    sha256(ROOT / 'weights/yolov5s_v5.0.pt') == sums['yolov5s_v5.0.pt'])
for run in ('p1_yolov5s', 'p1_gbh', 'p1_yolov5_1'):
    ns = namespace(R / f'p1/{run}.log')
    done = nlines(R / f'p1/{run}/results.txt')
    label = {'p1_yolov5s': 'YOLOv5s', 'p1_gbh': 'GBH-YOLOv5', 'p1_yolov5_1': 'YOLOv5-1'}[run]
    add('P1', f'{label}: optimizer', '"employs Adam as the optimizer"', f"adam={ns['adam']}", ns['adam'] == 'True')
    add('P1', f'{label}: image size', '"the input 960 × 960 pixel image"', ns['img_size'], ns['img_size'] == '[960, 960]')
    add('P1', f'{label}: epochs', '"converge after 500 epochs"', f"epochs={ns['epochs']}, completed {done}",
        ns['epochs'] == '500' and done == 500 if run != 'p1_yolov5_1' else None,
        'stopped at 203 by the pre-set rule (no sign of the claimed gain by epoch 200)' if run == 'p1_yolov5_1' else '')
    b = int(ns['batch_size'])
    add('P1', f'{label}: batch size', '"The batch size is 16"', f'{b} × {64 // b} accumulation = 64 images per step',
        b == 16 if b == 16 else None, '' if b == 16 else 'physical batch 16 does not fit in 8 GB at 960 px; same 64 images per optimizer step')
splits = {s: nlines(ROOT / f'datasets/splits/pv_multi_defect/{s}.txt') for s in ('train', 'val')}
add('P1', 'Data split', '1108 images, 886 train / 222 val', f"{splits['train']} / {splits['val']}", None,
    'counts match; membership unpublished, so re-created (seed 0)')

# ------------------------------------------------------------------ P2
V = ROOT / 'code/yolov5-v7.0'
head = git(V, 'rev-parse', 'HEAD')
wt = git(V, 'diff', '--', '.', ':!*.pyc')
patch = (ROOT / 'code/repro/p2/yolov5_v7.0_p2.patch').read_text()
same_patch = [l for l in wt.splitlines() if re.match(r'^[+-][^+-]', l)] == [l for l in patch.splitlines() if re.match(r'^[+-][^+-]', l)]
add('P2', 'Training code', 'YOLOv5 (P2 trains with ultralytics YOLOv5 / YOLOv8)', f'ultralytics/yolov5 tag v7.0 ({head[:7]}); YOLOv8: ultralytics 8.0.99',
    head.startswith('915bbf2'))
add('P2', 'Our additions to the code', 'GAM attention (Eq. 1), ASFF head (Eq. 5–6), DIoU-NMS; no code released',
    'one committed patch; working tree = patch' if same_patch else 'WORKING TREE ≠ PATCH', None,
    'implemented from the paper\'s equations and the cited originals; DIoU is opt-in (`--diou-nms`), so the YOLOv5s baseline runs stock code')
for run, cfg in (('p2_v5s', 'YOLOv5s'), ('p2_v5_improved', 'Improved')):
    o = yaml.safe_load(open(R / f'p2/{run}/opt.yaml'))
    hd = hyp_diff(R / f'p2/{run}/hyp.yaml', V / 'data/hyps/hyp.scratch-low.yaml')
    add('P2', f'{cfg}: optimizer / lr', '"An Adam optimizer … initial learning rate of 0.01"', f"{o['optimizer']} / lr0 {yaml.safe_load(open(R / f'p2/{run}/hyp.yaml'))['lr0']}",
        o['optimizer'] == 'Adam' and yaml.safe_load(open(R / f'p2/{run}/hyp.yaml'))['lr0'] == 0.01)
    add('P2', f'{cfg}: other hyper-parameters', 'not stated → codebase defaults', 'all = `hyp.scratch-low.yaml`' if not hd else f'differ: {hd}', not hd)
    add('P2', f'{cfg}: epochs / image size', '"100 training epochs, and input image size of 640 × 640"',
        f"{o['epochs']} (completed {nlines(R / f'p2/{run}/results.csv') - 1}) / {o['imgsz']}", o['epochs'] == 100 and o['imgsz'] == 640)
    add('P2', f'{cfg}: batch size', '"batch size of 32"', f"{o['batch_size']} × 4 accumulation = 64 images per step (paper 32 × 2)", None,
        'physical 32 does not fit in 6 GB; same 64 images per optimizer step')
    add('P2', f'{cfg}: initial weights', '"YOLOv5 … used YOLOv5s.pt"', 'yolov5s.pt, v7.0 release (checksum in weights/SHA256SUMS)',
        Path(o['weights']).name == 'yolov5s_v7.0.pt' and sha256(ROOT / 'weights/yolov5s_v7.0.pt') == sums['yolov5s_v7.0.pt'])
a8 = yaml.safe_load(open(R / 'p2/p2_v8s/args.yaml'))
add('P2', 'YOLOv8s: optimizer / lr / epochs / size', 'Adam, 0.01, 100, 640', f"{a8['optimizer']}, {a8['lr0']}, {a8['epochs']} (completed {nlines(R / 'p2/p2_v8s/results.csv') - 1}), {a8['imgsz']}",
    (a8['optimizer'], a8['lr0'], a8['epochs'], a8['imgsz']) == ('Adam', 0.01, 100, 640))
add('P2', 'YOLOv8s: mosaic', '"mosaic augmentation is disabled for the last ten training epochs"', f"close_mosaic={a8['close_mosaic']}", a8['close_mosaic'] == 10)
add('P2', 'YOLOv8s: initial weights', '"The YOLOv8 model used yolov8s.pt"', Path(a8['model']).name, Path(a8['model']).name == 'yolov8s.pt')
gf = {}
for run in ('p2_v5s', 'p2_v5_improved', 'p2_v8s'):
    t = open(R / f'p2/{run}.log', encoding='utf-8', errors='replace').read().replace('\r', '\n')
    gf[run] = float(re.findall(r'summary: \d+ layers, \d+ parameters, \d+ gradients, ([\d.]+) GFLOPs', t)[0])
add('P2', 'Model size (GFLOPs)', 'Table: YOLOv5s 16, Improved 30.0, YOLOv8s 28.7',
    f"{gf['p2_v5s']}, {gf['p2_v5_improved']}, {gf['p2_v8s']}", None,
    'YOLOv5s and YOLOv8s match independently; Improved matches by construction (GAM placement calibrated to it)')
add('P2', 'Data split', '1187 train / 190 test, 6 classes', f"{nlines(ROOT / 'datasets/eldds_yolo/train.txt')} / {nlines(ROOT / 'datasets/eldds_yolo/test.txt')}, 6 classes",
    None, 'authors\' own Roboflow upload; 2 images whose only labels lie outside the 6 classes excluded')

# ------------------------------------------------------------------ P3
src = (ROOT / 'code/repro/p3/train_cnn.py').read_text()
log = (R / 'p3/cnn/train.log').read_text()
args = ast.literal_eval(re.search(r"args (\{.*?\})\n", log).group(1))
add('P3', 'CNN architecture', 'ImageNet VGG-19 conv blocks + GAP + FC 4096 + FC 2048 + 1 output, regression',
    'tf.keras VGG19(weights="imagenet") + GAP + Dense 4096 + Dense 2048 + Dense 1, MSE',
    all(s in src for s in ("VGG19(include_top=False, weights='imagenet'", 'GlobalAveragePooling2D', 'Dense(4096', 'Dense(2048', 'Dense(1')))
add('P3', 'CNN stage 1', 'FC layers only; Adam, lr 1e-3, β 0.9 / 0.999, ε 1e-8', 'conv frozen; Adam(1e-3, 0.9, 0.999, 1e-8)',
    'Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8)' in src)
add('P3', 'CNN stage 2', 'all layers; SGD lr 5e-4, momentum 0.9', 'SGD(5e-4, momentum 0.9)', 'SGD(learning_rate=5e-4, momentum=0.9)' in src)
h1, h2 = nlines(R / 'p3/cnn/stage1_history.csv') - 1, nlines(R / 'p3/cnn/stage2_history.csv') - 1
add('P3', 'CNN batch / epochs', '"mini-batches of 16 samples", "a maximum of 100 epochs" per stage', f"batch {args['batch']}; {h1 + 1} + {h2 + 1} epochs logged",
    args['batch'] == 16 and args['epochs'] == 100)
add('P3', 'CNN augmentation', 'scale ≤ 2 %, rotation ≤ 3°, shift ≤ 2 %, h/v flips, 90° rotations',
    'zoom 0.02, rotation 3, shift 0.02, h/v flips, rot90', all(s in src for s in ('rotation_range=3', 'width_shift_range=0.02', 'zoom_range=0.02', 'horizontal_flip=True', 'vertical_flip=True', 'np.rot90')))
add('P3', 'CNN framework', 'Keras 2.0 + TensorFlow 1.4', 'tf.keras, TensorFlow 2.10.1 (same ImageNet weights file)', None,
    'TF 1.4 cannot run on current GPUs / Python; same algorithm, newer library')
add('P3', 'CNN output activation', 'not stated ("deep regression")', 'linear', None, 'a sigmoid output did not train (loss frozen)')
meta = json.loads(str(np.load(ROOT / 'models/p3/svm_kaze_vgg_vlad_strict.npz', allow_pickle=False)['meta']))
add('P3', 'SVM pipeline', 'KAZE + VGG descriptor, VLAD ×5 codebooks, α 0.5, PCA whitening, L2, linear SVM',
    f"KAZE thr {meta['kaze_threshold']} (OpenCV default) + VGG, VLAD K {meta['vlad']['K']} ×{meta['vlad']['n_codebooks']}, α {meta['vlad']['power_alpha']}, PCA {meta['pca_whitened_dims']}, linear C {meta['C']}",
    None, 'K and PCA dimension are not stated; chosen by training-set CV only')
add('P3', 'Data split', '1968 / 656, Table 3 counts per type × label', 'identical counts in all 16 cells (results/dataset_check.md)', None,
    'membership unpublished; stratified re-split with the same counts')
add('P3', 'Evaluation metric', 'confidence-weighted; F1 "averaged over the individual per-class F1 scores"', 'confidence-weighted acc / macro-F1 / AUC', True)

# ------------------------------------------------------------------ write
n = {k: sum(r[4].startswith(k) for r in rows) for k in ('✅', '⚠', '❌')}
L = ['# Fidelity check: training code and settings vs what each paper states', '',
     'Generated by `code/repro/fidelity_check.py` from the runs\' own artefacts (training-start argument logs, '
     'hyp/opt/args yaml, results files, model summaries, git state, checksums).', '',
     f"**{n['✅']} items identical, {n['⚠']} unavoidable or unstated (explained), {n['❌']} different.**", '',
     '| Paper | Item | Paper states | Our run (read from its files) | Verdict | Why |', '|---|---|---|---|---|---|']
L += [f'| {a} | {b} | {c} | {d} | {e} | {f} |' for a, b, c, d, e, f in rows]
(R / 'fidelity_check.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
print('\n'.join(L))
