"""P3 SVM (Deitsch et al. 2019, Sec. 3.1): best pipeline KAZE/VGG + VLAD + linear SVM, on ELPV.

Paper -> here:
- Keypoints: KAZE (OpenCV defaults), descriptors: VGG (xfeatures2d, 120-dim real-valued variant = default VGG_120,
  OpenCV defaults). Best configuration: with sample weighting, without masking, linear SVM (Sec. 4.3).
- Encoding: VLAD, dictionary by mini-batch k-means on a random subset of training descriptors; residual sums,
  power normalisation alpha = 0.5, L2. Five VLADs from different training subsets / random seeds, concatenated,
  jointly PCA-whitened, L2-normalised. Codebook size K, subset size and PCA output dimension are NOT given in the
  paper: K = 100, 100k descriptors per codebook; the PCA dimension is selected together with C by the inner CV from
  {32, 64, 128, 256, 512} (full-rank whitening of 1968 samples in 60000-d drowns the signal: inner-CV F1 ~0.5).
- Linear SVM (LIBLINEAR via scikit-learn LinearSVC): L2 penalty, squared hinge loss, C from {10^k | k=-2..6} chosen by
  average (macro) F1 in an inner stratified 5-fold CV on the training set (grid search).
- Weights: confidence weights w (Table 2) x class-balancing weights c_j (Eq. 5).
- Features (VLAD + PCA) are fitted once on the full training set (unsupervised); the inner CV selects (PCA dim, C).

usage (repo root): envs/p3svm/Scripts/python.exe -I code/repro/p3/train_svm.py --out results/p3/svm [--jobs 6]
"""
import argparse
import csv
import json
import os
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
from elpv_common import class_weights, confidence_weight, fmt_metrics, load_split, metrics  # noqa: E402

K, N_VLAD, SUBSET, ALPHA = 100, 5, 100_000, 0.5
PCA_DIMS = [32, 64, 128, 256, 512]
C_GRID = [10.0 ** k for k in range(-2, 7)]


def describe(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    kps = cv2.KAZE_create().detect(img, None)
    kps, desc = cv2.xfeatures2d.VGG_create().compute(img, kps)
    return np.zeros((0, 120), np.float32) if desc is None else desc.astype(np.float32)


def vlad(desc, km):
    v = np.zeros((K, desc.shape[1] if len(desc) else 120), np.float32)
    if len(desc):
        a = km.predict(desc)
        for k in np.unique(a):
            v[k] = (desc[a == k] - km.cluster_centers_[k]).sum(0)
    v = v.ravel()
    v = np.sign(v) * np.abs(v) ** ALPHA
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n > 0, n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / 'train.log', 'a', buffering=1)

    def say(*m):
        s = time.strftime('[%F %T] ') + ' '.join(map(str, m))
        print(s, flush=True)
        log.write(s + '\n')

    say('opencv', cv2.__version__, '| K', K, 'N_VLAD', N_VLAD, 'subset', SUBSET)
    tr_paths, tr_p, tr_t = load_split(a.root, 'train')
    te_paths, te_p, te_t = load_split(a.root, 'test')

    cache = out / 'descriptors.npz'
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        Dtr, Dte = list(z['tr']), list(z['te'])
        say('loaded cached descriptors')
    else:
        t0 = time.time()
        Dtr = Parallel(n_jobs=a.jobs)(delayed(describe)(p) for p in tr_paths)
        Dte = Parallel(n_jobs=a.jobs)(delayed(describe)(p) for p in te_paths)
        np.savez(cache, tr=np.array(Dtr, dtype=object), te=np.array(Dte, dtype=object))
        say(f'KAZE/VGG descriptors in {time.time() - t0:.0f}s')
    nk = np.array([len(d) for d in Dtr])
    say(f'keypoints/cell train: mean {nk.mean():.0f}, min {nk.min()}, max {nk.max()}, zero {int((nk == 0).sum())}')

    allD = np.concatenate([d for d in Dtr if len(d)])
    Vtr, Vte = [], []
    for s in range(N_VLAD):
        rs = np.random.RandomState(s)
        sub = allD[rs.choice(len(allD), min(SUBSET, len(allD)), replace=False)]
        km = MiniBatchKMeans(n_clusters=K, random_state=s).fit(sub)
        Vtr.append(np.stack([vlad(d, km) for d in Dtr]))
        Vte.append(np.stack([vlad(d, km) for d in Dte]))
        say(f'VLAD {s + 1}/{N_VLAD} done')
    Vtr, Vte = np.hstack(Vtr), np.hstack(Vte)
    pca = PCA(n_components=max(PCA_DIMS), whiten=True, random_state=0).fit(Vtr)
    Ptr, Pte = pca.transform(Vtr), pca.transform(Vte)  # whitened components are nested: first d columns = PCA(d)
    say(f'VLAD dim {Vtr.shape[1]} -> PCA-whitened up to {Ptr.shape[1]}')

    y = (tr_p > 0).astype(int)
    cw = class_weights(tr_p)
    w = confidence_weight(tr_p)
    swt = w * np.where(y == 1, cw[1], cw[0])
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    cv = {}
    for d in PCA_DIMS:
        Xtr = l2(Ptr[:, :d])
        for C in C_GRID:
            f1s = []
            for itr, iva in skf.split(Xtr, y):
                clf = LinearSVC(C=C, penalty='l2', loss='squared_hinge', max_iter=20000, random_state=0)
                clf.fit(Xtr[itr], y[itr], sample_weight=swt[itr])
                f1s.append(f1_score(y[iva], clf.predict(Xtr[iva]), average='macro', sample_weight=w[iva]))
            cv[(d, C)] = float(np.mean(f1s))
            say(f'PCA {d:4d}  C={C:g}: inner-CV macro F1 (weighted) {cv[(d, C)]:.4f}')
    d, C = max(cv, key=cv.get)
    Xtr, Xte = l2(Ptr[:, :d]), l2(Pte[:, :d])
    clf = LinearSVC(C=C, penalty='l2', loss='squared_hinge', max_iter=20000, random_state=0).fit(Xtr, y, sample_weight=swt)
    score = clf.decision_function(Xte)
    m = metrics(te_p, score, te_t, threshold=0.0)
    say(f'selected PCA dim {d}, C={C:g} (inner-CV F1 {cv[(d, C)]:.4f})\nTEST\n' + fmt_metrics(m))
    json.dump({'pca_dim': d, 'C': C, 'cv_f1': {f'{k[0]}|{k[1]:g}': v for k, v in cv.items()}, 'test': m},
              open(out / 'test_metrics.json', 'w'), indent=1)
    with open(out / 'test_predictions.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['path', 'p_true', 'type', 'svm_decision'])
        wr.writerows(zip([os.path.basename(p) for p in te_paths], te_p, te_t, score))


if __name__ == '__main__':
    main()
