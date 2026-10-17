"""Peak VRAM and s/iter of a training step (AMP fwd+bwd) per batch size, for picking the physical batch.

On Windows the driver spills into system RAM instead of raising OOM, so a batch is "fitting" only if
peak reserved memory stays below the card's VRAM (minus ~0.4 GB for context) and s/iter stays sane.
usage (from the codebase root): python ../repro/mem_probe.py --cfg X.yaml --img 960 --batches 2 4 8 16 [--v8 yolov8s.pt]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import torch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--cfg')
ap.add_argument('--v8')
ap.add_argument('--img', type=int, required=True)
ap.add_argument('--nc', type=int, default=6)
ap.add_argument('--batches', type=int, nargs='+', required=True)
ap.add_argument('--iters', type=int, default=6)
a = ap.parse_args()
torch.backends.cudnn.benchmark = True
total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
if a.v8:
    from ultralytics.nn.tasks import DetectionModel
    import yaml
    m = DetectionModel(torch.load(a.v8, map_location='cpu')['model'].yaml, nc=a.nc, verbose=False)
else:
    try:
        from models.yolo import Model as M
    except ImportError:
        from models.yolo import DetectionModel as M
    m = M(a.cfg, ch=3, nc=a.nc)
m = m.cuda().train()
opt = torch.optim.Adam(m.parameters(), 1e-3)
scaler = torch.cuda.amp.GradScaler()
print(f'card {total:.2f} GiB')
for b in a.batches:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    x = torch.rand(b, 3, a.img, a.img, device='cuda')
    ts = []
    for i in range(a.iters):
        torch.cuda.synchronize(); t = time.time()
        with torch.cuda.amp.autocast():
            out = m(x)
            out = out if isinstance(out, (list, tuple)) else [out]
            loss = sum(o.float().mean() for o in out if torch.is_tensor(o))
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad()
        torch.cuda.synchronize(); ts.append(time.time() - t)
    r = torch.cuda.max_memory_reserved() / 2 ** 30
    st = sorted(ts[2:])[len(ts[2:]) // 2]
    print(f'batch {b:3d}: peak reserved {r:5.2f} GiB  median {st:6.3f} s/iter  first {ts[0]:6.1f} s  '
          f'{"OK" if r < total - 0.4 else "SPILL"}', flush=True)
    del x
    if r > total + 0.5 or st > 30:
        break
