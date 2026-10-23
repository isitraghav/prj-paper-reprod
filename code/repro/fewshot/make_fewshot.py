"""Few-shot transfer: how many labelled target images does a model from another PV paper need, and does starting from
it beat starting from COCO?

  pv:    P2's ELDDS-trained YOLOv5s (models/p2/yolov5s.pt) vs COCO yolov5s v7.0 -> PV-Multi-Defect (P1 data),
         P2's codebase and recipe (yolov5 v7.0, Adam lr 0.01, hyp.scratch-low), scored on PV val 222.
  eldds: P1's PV-trained YOLOv5s (models/p1/yolov5s.pt) vs COCO yolov5s v5.0 -> ELDDS1400C5 (P2 data),
         P1's codebase and recipe (GBH fork, Adam lr 0.001, hyp.gbh), scored on ELDDS test 190.
n in {10, 50, 100} target training images (nested random subsets, seed 0). 640 px, batch 16, ~1500 iterations per
run (epochs = ceil(1500 / iterations per epoch); the GBH fork warms up for 1000 iterations), validation at the end only.

Writes datasets/splits/fewshot/{pv,eldds}_n{N}.txt (file names), datasets/fewshot/*.txt + *.yaml (absolute paths),
and prints the queue lines (results/queue/jobs.txt format).

usage (repo root): envs/p3cnn/Scripts/python.exe -I code/repro/fewshot/make_fewshot.py
"""
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT.as_posix()
NS, ITERS, BATCH = (10, 50, 100), 1500, 16
SPL, DS = ROOT / 'datasets/splits/fewshot', ROOT / 'datasets/fewshot'
SPL.mkdir(parents=True, exist_ok=True)
DS.mkdir(parents=True, exist_ok=True)
SRC = {'pv': (ROOT / 'datasets/pv_multi_defect_yolo/train.txt', ROOT / 'datasets/pv_multi_defect_yolo/val.txt',
              5, "['broken', 'hot_spot', 'black_border', 'scratch', 'no_electricity']"),
       'eldds': (ROOT / 'datasets/eldds_yolo/train.txt', ROOT / 'datasets/eldds_yolo/test.txt',
                 6, "['Examined', 'ShortCircuitString', 'ShortCircuitCell-LowPowerCell', 'Crack', 'MicroCrack', 'OtherError']")}
# the GBH fork cannot read the numpy-format label cache that yolov5 v7.0 wrote next to eldds_yolo/test.txt
# (torch.load -> UnpicklingError), so the ELDDS runs score on a copy of the list with its own cache
(DS / 'eldds_test.txt').write_text(open(SRC['eldds'][1]).read())
SRC['eldds'] = (SRC['eldds'][0], DS / 'eldds_test.txt') + SRC['eldds'][2:]
jobs = []
for ds, (train, val, nc, names) in SRC.items():
    pool = sorted(l.strip() for l in open(train) if l.strip())
    order = random.Random(0).sample(pool, len(pool))  # nested: the first n of one fixed permutation
    for n in NS:
        sub = sorted(order[:n])
        (SPL / f'{ds}_n{n}.txt').write_text(''.join(os.path.basename(s) + '\n' for s in sub))
        (DS / f'{ds}_n{n}.txt').write_text(''.join(s + '\n' for s in sub))
        y = DS / f'{ds}_n{n}.yaml'
        y.write_text(f'train: {(DS / f"{ds}_n{n}.txt").as_posix()}\nval: {Path(val).as_posix()}\nnc: {nc}\nnames: {names}\n')
        ep = math.ceil(ITERS / math.ceil(n / BATCH))
        for init in ('src', 'coco'):
            name = f'fs_{ds}_n{n}_{init}'
            if ds == 'pv':  # P2's model and codebase -> P1 data
                w = f'{P}/models/p2/yolov5s.pt' if init == 'src' else f'{P}/weights/yolov5s_v7.0.pt'
                cmd = (f'{P}/envs/p2/Scripts/python.exe train.py --cfg {P}/code/repro/p2/yolov5s.yaml --weights {w} '
                       f'--data {y.as_posix()} --hyp data/hyps/hyp.scratch-low.yaml --optimizer Adam --epochs {ep} '
                       f'--batch-size {BATCH} --img 640 --seed 0 --workers 1 --noval '
                       f'--project {P}/results/fewshot --name {name} --exist-ok')
                wd = 'code/yolov5-v7.0'
            else:  # P1's model and codebase -> P2 data
                w = f'{P}/models/p1/yolov5s.pt' if init == 'src' else f'{P}/weights/yolov5s_v5.0.pt'
                cmd = (f'{P}/envs/p1/Scripts/python.exe train.py --cfg {P}/code/repro/p1/yolov5s.yaml --weights {w} '
                       f'--data {y.as_posix()} --hyp {P}/code/repro/p1/hyp.gbh.yaml --adam --epochs {ep} '
                       f'--batch-size {BATCH} --img-size 640 640 --workers 0 --device 0 --notest '
                       f'--project {P}/results/fewshot --name {name} --exist-ok')
                wd = 'code/gbh-yolov5'
            jobs.append(f'{name} | {wd} | {cmd} | results/fewshot/{name}.log')
sys.stdout.write('\n'.join(jobs) + '\n')
