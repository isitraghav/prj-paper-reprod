"""Detection metrics shared by cross_eval.py and sensitivity.py: numpy re-implementation of yolov5 v7.0
utils/metrics.py (ap_per_class, 101-point AP, P/R at max smoothed mean F1) and val.py process_batch matching."""
import numpy as np

IOUV = np.linspace(0.5, 0.95, 10)


def box_iou(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def process_batch(det, lab):
    """det (n,6) [cls,conf,x1..y2], lab (m,5) [cls,x1..y2] -> correct (n,10) bool (val.py logic)."""
    correct = np.zeros((len(det), len(IOUV)), bool)
    if not len(det) or not len(lab):
        return correct
    iou = box_iou(lab[:, 1:], det[:, 2:])
    cls_ok = lab[:, 0:1] == det[:, 0]
    for i, t in enumerate(IOUV):
        x = np.nonzero((iou >= t) & cls_ok)
        if x[0].shape[0]:
            m = np.concatenate((np.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
            if x[0].shape[0] > 1:
                m = m[m[:, 2].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), i] = True
    return correct


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return np.trapz(np.interp(x, mrec, mpre), x)


def smooth(y, f=0.05):
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')


def ap_per_class(tp, conf, pred_cls, target_cls, curve=False):
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    classes = np.unique(target_cls)
    px = np.linspace(0, 1, 1000)
    ap, p, r = np.zeros((len(classes), tp.shape[1])), np.zeros((len(classes), 1000)), np.zeros((len(classes), 1000))
    pr_curve = None
    for ci, c in enumerate(classes):
        k = pred_cls == c
        n_l, n_p = (target_cls == c).sum(), k.sum()
        if n_p == 0 or n_l == 0:
            continue
        fpc, tpc = (1 - tp[k]).cumsum(0), tp[k].cumsum(0)
        recall = tpc / (n_l + 1e-16)
        precision = tpc / (tpc + fpc)
        r[ci] = np.interp(-px, -conf[k], recall[:, 0], left=0)
        p[ci] = np.interp(-px, -conf[k], precision[:, 0], left=1)
        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])
        if curve and len(classes) == 1:
            mrec = np.concatenate(([0.0], recall[:, 0], [1.0]))
            mpre = np.flip(np.maximum.accumulate(np.flip(np.concatenate(([1.0], precision[:, 0], [0.0])))))
            pr_curve = (mrec, mpre)
    f1 = 2 * p * r / (p + r + 1e-16)
    i = smooth(f1.mean(0), 0.1).argmax()
    return dict(P=float(p[:, i].mean()), R=float(r[:, i].mean()), mAP50=float(ap[:, 0].mean()),
                mAP=float(ap.mean()), ap50_per_class={int(c): float(a) for c, a in zip(classes, ap[:, 0])},
                pr_curve=pr_curve)


def nms_agnostic(d, iou_thr, max_det=300):
    d = d[np.argsort(-d[:, 1])]
    keep, boxes = [], d[:, 2:]
    alive = np.ones(len(d), bool)
    for i in range(len(d)):
        if not alive[i]:
            continue
        keep.append(i)
        if len(keep) == max_det:
            break
        alive[i + 1:] &= box_iou(boxes[i:i + 1], boxes[i + 1:])[0] <= iou_thr
    return d[keep]


def det_stats(preds, gts, keys):
    """Per-image correct matrices -> concatenated arrays for ap_per_class."""
    per = {}
    for k in keys:
        d, g = preds[k], gts[k]
        per[k] = (process_batch(d, g), d[:, 1], d[:, 0], g[:, 0])
    return per


def det_metrics(per, keys, curve=False):
    tp = np.concatenate([per[k][0] for k in keys])
    conf = np.concatenate([per[k][1] for k in keys])
    pc = np.concatenate([per[k][2] for k in keys])
    tc = np.concatenate([per[k][3] for k in keys])
    return ap_per_class(tp, conf, pc, tc, curve=curve)


def ci(vals):
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
