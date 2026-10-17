"""Shared P3 helpers: ELPV loading, the paper's labels/weights, and evaluation metrics.

Labels (paper Table 2): p = 0 -> functional; p in {1/3, 2/3, 1} -> defective (non-confident ratings are labelled
defective). Confidence weights w: p=0 -> 1, p=1/3 -> 0.33, p=2/3 -> 0.67, p=1 -> 1 ("Non-confident decisions obtain a
weight lower than 100% for the evaluation of the classifier performance").
Class balancing (paper Eq. 5, King & Zeng 2001): c_j = S / (2 n_j) over the training set, j = functional/defective.
"""
from pathlib import Path

import numpy as np

DATA = 'datasets/raw/elpv-dataset/src/elpv_dataset/data'
SPLITS = 'datasets/splits/elpv'


def confidence_weight(p):
    return np.where(np.isclose(p, 1 / 3, atol=1e-3), 0.33, np.where(np.isclose(p, 2 / 3, atol=1e-3), 0.67, 1.0))


def load_split(root, split):
    """-> paths (list), p (float array), types (array of 'mono'/'poly')"""
    root = Path(root)
    lab = {}
    for l in (root / DATA / 'labels.csv').read_text().splitlines():
        if l.strip():
            path, p, t = l.split()
            lab[path] = (float(p), t)
    paths = (root / SPLITS / f'{split}.txt').read_text().split()
    p = np.array([lab[x][0] for x in paths])
    t = np.array([lab[x][1] for x in paths])
    return [str(root / DATA / x) for x in paths], p, t


def class_weights(p_train):
    y = (p_train > 0).astype(int)
    S, n = len(y), np.bincount(y, minlength=2)
    return S / (2.0 * n)  # [c_functional, c_defective]


def metrics(p_true, score, types=None, threshold=0.5):
    """Binary evaluation as in the paper: ground truth defective iff p > 0, prediction defective iff score >= threshold
    (CNN: probability thresholded at 50%; SVM: decision value >= 0 via threshold=0). Reports confidence-weighted
    (primary, Table 2 caption) and unweighted accuracy / macro-F1, plus ROC AUC. Returned per subset all/mono/poly."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    p_true, score = np.asarray(p_true), np.asarray(score)
    y, yhat, w = (p_true > 0).astype(int), (score >= threshold).astype(int), confidence_weight(p_true)
    out = {}
    subsets = {'all': np.ones(len(y), bool)}
    if types is not None:
        subsets.update({t: np.asarray(types) == t for t in ('mono', 'poly')})
    for name, m in subsets.items():
        out[name] = {
            'n': int(m.sum()),
            'acc_weighted': accuracy_score(y[m], yhat[m], sample_weight=w[m]),
            'f1_macro_weighted': f1_score(y[m], yhat[m], average='macro', sample_weight=w[m]),
            'acc': accuracy_score(y[m], yhat[m]),
            'f1_macro': f1_score(y[m], yhat[m], average='macro'),
            'auc_weighted': roc_auc_score(y[m], score[m], sample_weight=w[m]),
            'auc': roc_auc_score(y[m], score[m]),
        }
    return out


def fmt_metrics(m):
    return '\n'.join(f"{k:5s} n={v['n']:4d}  acc_w {100 * v['acc_weighted']:6.2f}  F1_w {100 * v['f1_macro_weighted']:6.2f}  "
                     f"acc {100 * v['acc']:6.2f}  F1 {100 * v['f1_macro']:6.2f}  AUC_w {100 * v['auc_weighted']:6.2f}  "
                     f"AUC {100 * v['auc']:6.2f}" for k, v in m.items())
