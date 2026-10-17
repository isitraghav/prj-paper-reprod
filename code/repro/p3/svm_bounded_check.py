"""P3 SVM bounded check (user-approved 2026-10-18): is the gap to the paper (acc 69.70 % vs 82.44 %) caused by
unstated implementation details? Fixed, pre-declared grid; selection by inner 5-fold CV on the TRAINING split only;
the selected configuration is evaluated on the test split exactly once. The strict-defaults result stays the
reproduction number; this one is reported separately as "better-specified" with its deviations.

Grid (everything else as train_svm.py: VGG descriptor defaults, VLAD K=100 x5 seeds, linear SVM, weights):
  KAZE threshold   : 0.001 (OpenCV default, paper), 0.0003, 0.0001       -> denser keypoints
  encoding         : pcaw{32,64,128,256} (current), nopca, regw{0.1,1} (full-rank whitening, lambda + r*mean(lambda))
  C                : 10^-2 .. 10^2

usage (repo root): envs/p3svm/Scripts/python.exe -I code/repro/p3/svm_bounded_check.py --out results/p3/svm_check
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_svm as T  # noqa: E402  (vlad, l2, K, constants)
from elpv_common import class_weights, confidence_weight, fmt_metrics, load_split, metrics  # noqa: E402

THRESHOLDS = [0.001, 0.0003, 0.0001]
ENCODINGS = ['pcaw32', 'pcaw64', 'pcaw128', 'pcaw256', 'nopca', 'regw0.1', 'regw1']
C_GRID = [10.0 ** k for k in range(-2, 3)]


def describe(path, thr):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    kps = cv2.KAZE_create(threshold=thr).detect(img, None)
    kps, desc = cv2.xfeatures2d.VGG_create().compute(img, kps)
    return np.zeros((0, 120), np.float32) if desc is None else desc.astype(np.float32)


def vlads(Dtr, Dte):
    allD = np.concatenate([d for d in Dtr if len(d)])
    Vtr, Vte = [], []
    for s in range(T.N_VLAD):
        rs = np.random.RandomState(s)
        km = MiniBatchKMeans(n_clusters=T.K, random_state=s).fit(allD[rs.choice(len(allD), min(T.SUBSET, len(allD)), replace=False)])
        Vtr.append(np.stack([T.vlad(d, km) for d in Dtr]).astype(np.float32))
        Vte.append(np.stack([T.vlad(d, km) for d in Dte]).astype(np.float32))
    return np.hstack(Vtr), np.hstack(Vte)


class GramPCA:
    """PCA of the training VLADs via the n x n Gram matrix (n = 1968 << D = 60000): same components, eigenvalues
    and projections as a full-SVD PCA, at a fraction of the memory (runs next to the CNN)."""

    def __init__(self, V):
        self.mean = V.mean(0)
        self.Xc = V - self.mean
        mu, U = np.linalg.eigh((self.Xc @ self.Xc.T).astype(np.float64))
        keep = mu > 1e-10 * mu.max()
        order = np.argsort(mu[keep])[::-1]
        self.mu, self.U = mu[keep][order], U[:, keep][:, order]
        self.explained_variance_ = self.mu / (len(V) - 1)

    def transform_train(self):
        return self.U * np.sqrt(self.mu)  # = Xc @ components.T

    def transform(self, V):
        return ((V - self.mean) @ self.Xc.T).astype(np.float64) @ (self.U / np.sqrt(self.mu))


def encode(name, Vtr, Vte, pca):
    if name == 'nopca':
        return T.l2(Vtr), T.l2(Vte)
    Ptr, Pte = pca.transform_train(), pca.transform(Vte)  # PCA without whitening, all components
    lam = pca.explained_variance_
    if name.startswith('pcaw'):
        d = int(name[4:])
        s = 1.0 / np.sqrt(lam[:d])
        return T.l2(Ptr[:, :d] * s), T.l2(Pte[:, :d] * s)
    r = float(name[4:])
    s = 1.0 / np.sqrt(lam + r * lam.mean())
    return T.l2(Ptr * s), T.l2(Pte * s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / 'check.log', 'a', buffering=1)

    def say(*m):
        s = time.strftime('[%F %T] ') + ' '.join(map(str, m))
        print(s, flush=True)
        log.write(s + '\n')

    tr_paths, tr_p, tr_t = load_split(a.root, 'train')
    te_paths, te_p, te_t = load_split(a.root, 'test')
    y = (tr_p > 0).astype(int)
    w = confidence_weight(tr_p)
    cw = class_weights(tr_p)
    swt = w * np.where(y == 1, cw[1], cw[0])
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    rows, best = [], None
    for thr in THRESHOLDS:
        cache = out / f'desc_thr{thr:g}.npz'
        if cache.exists():
            z = np.load(cache, allow_pickle=True)
            Dtr, Dte = list(z['tr']), list(z['te'])
        else:
            Dtr = Parallel(n_jobs=a.jobs)(delayed(describe)(p, thr) for p in tr_paths)
            Dte = Parallel(n_jobs=a.jobs)(delayed(describe)(p, thr) for p in te_paths)
            np.savez(cache, tr=np.array(Dtr, dtype=object), te=np.array(Dte, dtype=object))
        nk = np.array([len(d) for d in Dtr])
        say(f'thr {thr:g}: keypoints/cell mean {nk.mean():.0f} (mono {nk[tr_t == "mono"].mean():.0f}, '
            f'poly {nk[tr_t == "poly"].mean():.0f}), zero {int((nk == 0).sum())}')
        Vtr, Vte = vlads(Dtr, Dte)
        pca = GramPCA(Vtr)
        for enc in ENCODINGS:
            Xtr, _ = encode(enc, Vtr, Vte, pca)
            for C in C_GRID:
                f = np.mean([f1_score(y[iva], LinearSVC(C=C, max_iter=20000, random_state=0)
                                      .fit(Xtr[itr], y[itr], sample_weight=swt[itr]).predict(Xtr[iva]),
                                      average='macro', sample_weight=w[iva]) for itr, iva in skf.split(Xtr, y)])
                rows.append((thr, enc, C, f))
                if best is None or f > best[3]:
                    best = (thr, enc, C, f)
            say(f'thr {thr:g} {enc:8s}: best inner-CV F1 {max(r[3] for r in rows if r[0] == thr and r[1] == enc):.4f}')
        del Vtr, Vte, pca
    with open(out / 'cv_table.csv', 'w', newline='') as f:
        csv.writer(f).writerows([('kaze_threshold', 'encoding', 'C', 'inner_cv_f1_weighted')] + rows)
    thr, enc, C, f = best
    say(f'SELECTED (training CV only): KAZE thr {thr:g}, {enc}, C {C:g}, inner-CV F1 {f:.4f}')

    z = np.load(out / f'desc_thr{thr:g}.npz', allow_pickle=True)
    Vtr, Vte = vlads(list(z['tr']), list(z['te']))
    Xtr, Xte = encode(enc, Vtr, Vte, GramPCA(Vtr))
    clf = LinearSVC(C=C, max_iter=20000, random_state=0).fit(Xtr, y, sample_weight=swt)
    m = metrics(te_p, clf.decision_function(Xte), te_t, threshold=0.0)
    say('TEST (single evaluation of the selected config)\n' + fmt_metrics(m))
    json.dump({'selected': {'kaze_threshold': thr, 'encoding': enc, 'C': C, 'inner_cv_f1': f}, 'test': m},
              open(out / 'test_metrics.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
