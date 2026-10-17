"""Export the two P3 SVM models (strict defaults = train_svm.py, better-specified = svm_bounded_check.py selection) as
portable .npz files (numpy arrays only, no pickles): 5 VLAD codebooks, PCA (if any), linear SVM weights.
Each model is refit exactly as in its training script from the cached descriptors, then reloaded from the .npz and
re-scored with plain numpy; the test metrics must equal the logged ones.

usage (repo root): envs/p3svm/Scripts/python.exe -I code/repro/p3/export_svm.py --out models/p3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_svm as T  # noqa: E402
from elpv_common import class_weights, confidence_weight, fmt_metrics, load_split, metrics  # noqa: E402

CONFIGS = {  # name: (descriptor cache, KAZE threshold, PCA-whitened dims or None, C, logged test acc_w)
    'svm_kaze_vgg_vlad_strict': ('results/p3/svm/descriptors.npz', 0.001, 32, 0.1, 69.70),
    'svm_kaze_vgg_vlad_thr1e-4_nopca': ('results/p3/svm_check/desc_thr0.0001.npz', 0.0001, None, 1.0, 79.65),
}


def vlad_np(desc, centers):  # = train_svm.vlad with MiniBatchKMeans.predict replaced by nearest centre
    v = np.zeros_like(centers)
    if len(desc):
        a = ((desc[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)
        for k in np.unique(a):
            v[k] = (desc[a == k] - centers[k]).sum(0)
    v = v.ravel()
    v = np.sign(v) * np.abs(v) ** T.ALPHA
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def predict_npz(path, descs):
    z = np.load(path)
    V = np.hstack([np.stack([vlad_np(d, c) for d in descs]) for c in z['codebooks']])
    if z['pca_components'].size:
        V = (V - z['pca_mean']) @ z['pca_components'].T / z['pca_scale']
    return T.l2(V) @ z['svm_coef'] + z['svm_intercept']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr_paths, tr_p, tr_t = load_split(a.root, 'train')
    te_paths, te_p, te_t = load_split(a.root, 'test')
    y = (tr_p > 0).astype(int)
    cw = class_weights(tr_p)
    swt = confidence_weight(tr_p) * np.where(y == 1, cw[1], cw[0])
    for name, (cache, thr, dims, C, logged) in CONFIGS.items():
        z = np.load(Path(a.root) / cache, allow_pickle=True)
        Dtr, Dte = list(z['tr']), list(z['te'])
        allD = np.concatenate([d for d in Dtr if len(d)])
        books, Vtr = [], []
        for s in range(T.N_VLAD):
            rs = np.random.RandomState(s)
            km = MiniBatchKMeans(n_clusters=T.K, random_state=s).fit(allD[rs.choice(len(allD), min(T.SUBSET, len(allD)), replace=False)])
            books.append(km.cluster_centers_.astype(np.float32))
            Vtr.append(np.stack([T.vlad(d, km) for d in Dtr]))
        Vtr = np.hstack(Vtr)
        mean = comp = scale = np.zeros(0, np.float32)
        if dims:
            pca = PCA(n_components=max(T.PCA_DIMS), whiten=True, random_state=0).fit(Vtr)  # as train_svm.py
            mean, comp, scale = pca.mean_, pca.components_[:dims], np.sqrt(pca.explained_variance_[:dims])
            Xtr = T.l2(pca.transform(Vtr)[:, :dims])
        else:
            Xtr = T.l2(Vtr)
        clf = LinearSVC(C=C, max_iter=20000, random_state=0).fit(Xtr, y, sample_weight=swt)
        meta = {'kaze_threshold': thr, 'descriptor': 'OpenCV xfeatures2d VGG_create() defaults (120-d)',
                'vlad': {'K': T.K, 'n_codebooks': T.N_VLAD, 'power_alpha': T.ALPHA, 'l2': True},
                'pca_whitened_dims': dims, 'C': C, 'opencv': '3.4.18 contrib', 'sklearn': '0.22.2.post1',
                'decision': 'defective iff score > 0'}
        path = out / f'{name}.npz'
        np.savez_compressed(path, codebooks=np.stack(books), pca_mean=mean, pca_components=comp, pca_scale=scale,
                            svm_coef=clf.coef_[0], svm_intercept=clf.intercept_, meta=json.dumps(meta))
        m = metrics(te_p, predict_npz(path, Dte), te_t, threshold=0.0)
        print(name, '\n' + fmt_metrics(m))
        acc = 100 * m['all']['acc_weighted']
        assert abs(acc - logged) < 0.01, f'{name}: test acc_w {acc:.2f} != logged {logged}'
        (out / f'{name}.json').write_text(json.dumps({'meta': meta, 'test': m}, indent=1))


if __name__ == '__main__':
    main()
