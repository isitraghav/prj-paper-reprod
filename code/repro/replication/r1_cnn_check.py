"""Which of R1's two P3 CNN files (PyTorch state_dicts, no code shipped) produced its saved test outputs, and is the
other one usable? Rebuilds VGG-19 features + GAP + FC 4096 + FC 2048 + 1 (sigmoid), ImageNet normalisation; the
architecture follows the state_dict shapes. A file whose outputs match cnn_full.npz is the reported model; a file
that scores as well on R1's test cells as on its training cells was trained on them.

usage (repo root): envs/p1/Scripts/python.exe -I code/repro/replication/r1_cnn_check.py <R1 folder>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'datasets/raw/elpv-dataset/src/elpv_dataset/data'
R1 = Path(sys.argv[1])
sys.stdout.reconfigure(encoding='utf-8')


def spearman(a, b):  # no ties in continuous outputs
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def acc_w(y, yhat, w):
    return float((w * (y == yhat)).sum() / w.sum())


def auc_w(y, s, w):  # weighted Mann-Whitney, ties count 1/2
    pos, neg = y == 1, y == 0
    gt = (s[pos][:, None] > s[neg][None]) + 0.5 * (s[pos][:, None] == s[neg][None])
    return float((w[pos][:, None] * w[neg][None] * gt).sum() / (w[pos].sum() * w[neg].sum()))


rows = [l.split() for l in (DATA / 'labels.csv').read_text().splitlines() if l.strip()]
p = np.array([float(r[1]) for r in rows])
w = np.where(np.isclose(p, 1 / 3, atol=1e-3), 0.33, np.where(np.isclose(p, 2 / 3, atol=1e-3), 0.67, 1.0))
z = np.load(R1 / 'final_work/deitsch_cnn/cnn_full.npz', allow_pickle=False)
te = z['te']
tr = np.setdiff1d(np.arange(len(rows)), te)


class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.features = torchvision.models.vgg19().features
        s.head = nn.Sequential(nn.Flatten(), nn.Linear(512, 4096), nn.ReLU(), nn.Linear(4096, 2048), nn.ReLU(), nn.Linear(2048, 1))

    def forward(s, x):
        return torch.sigmoid(s.head(s.features(x).mean((2, 3), keepdim=True)))


mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
out = {}
for f in ('cnn_vgg19_paper.pt', 'cnn_vgg19.pt'):
    net = Net()
    net.load_state_dict(torch.load(R1 / 'deitsch2019' / f, map_location='cpu'))
    net.eval().to(dev)
    s = []
    with torch.no_grad():
        for i in range(0, len(rows), 32):
            g = [torch.from_numpy(np.asarray(Image.open(DATA / rows[j][0]).convert('L'), np.float32) / 255)[None]
                 for j in range(i, min(i + 32, len(rows)))]
            s.append(net(torch.stack([(x.expand(3, -1, -1) - mean) / std for x in g]).to(dev)).cpu().numpy().ravel())
    s = np.concatenate(s)
    y = (p > 0).astype(int)
    r = {'spearman_vs_saved_test_outputs': spearman(s[te], z['pred'])}
    for name, ix in (('r1_test', te), ('r1_train', tr)):
        r[name] = {'acc_w': acc_w(y[ix], (s[ix] >= 0.5).astype(int), w[ix]), 'auc_w': auc_w(y[ix], s[ix], w[ix])}
    out[f] = r
    print(f, json.dumps(r))
json.dump(out, open(ROOT / 'results/replication_r1/cnn_check.json', 'w'), indent=1)
