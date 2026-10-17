"""Kaggle notebook (script kernel): train P1 / P2 models at the papers' exact settings, one job per GPU.

Pushed by code/repro/kaggle/orchestrate.py with SESSION set below. Kaggle "T4 x2": each GPU runs its own job
list (CUDA_VISIBLE_DEVICES pinned), so no DataParallel and no batch splitting. Training only; evaluation
is done locally on the downloaded weights with the same locked envs.

Inputs (private Kaggle datasets attached by the orchestrator):
  bundle.tar.gz   code/repro, datasets/splits, weights/SHA256SUMS (no secrets)
  eldds.tar.gz    datasets/raw/ELDDS1400C5 (Roboflow v1 export; only for P2 sessions)
  resume.tar.gz   results/... of a previous session of this kernel (optional)
Outputs (/kaggle/working): results/<p1|p2_kaggle>/<job>/, <job>_train.log, setup logs, STATUS.json

P1 paper: Adam, lr 0.001, batch 16, 500 ep, 960, COCO yolov5s.pt v5.0; env torch 1.8.1+cu111 (uv.lock p1).
P2 paper: Adam, lr 0.01, batch 32, 100 ep, 640, yolov5s.pt v7.0 / yolov8s.pt, YOLOv8 close_mosaic 10;
          env py 3.8.8, torch 1.11.0+cu113, ultralytics 8.0.99 (uv.lock p2).
"""
import glob
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time

SESSION = '__SESSION__'
PLAN = {  # gpu index -> jobs run in order on that GPU
    's1': {0: ['p1_gbh'], 1: ['p1_yolov5s']},
    's2': {0: ['p2_v5_improved_b32', 'p2_v5s_b32'], 1: ['p2_v8s_b32']},
}[SESSION]
SESSION_LIMIT_H, SAFETY_H = 12.0, 0.6

T0 = time.time()
OUT, PRJ, ENVS = '/kaggle/working', '/tmp/prj', '/tmp/envs'
GBH_COMMIT = '51ee042edf913bc1a6c3f25b6f296d6f8d6ef079'
PV_COMMIT = '2c4df0c1477032d6b0dd9372a517cdbfd6a09e09'
V7_TAG = 'v7.0'
LOG = open(f'{OUT}/{SESSION}_kernel.log', 'a', buffering=1)
LOCK = threading.Lock()
JOBSTATE = {}


def log(*a):
    s = f"[{time.strftime('%F %T')} +{(time.time() - T0) / 3600:.2f}h] " + ' '.join(map(str, a))
    with LOCK:
        print(s, flush=True)
        LOG.write(s + '\n')


def sh(cmd, cwd=None, env=None, check=True, logfile=None, timeout=None):
    log('$', cmd if isinstance(cmd, str) else ' '.join(cmd))
    out = open(logfile, 'a') if logfile else None
    try:
        rc = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, env={**os.environ, **(env or {})},
                            stdout=out, stderr=subprocess.STDOUT if out else None, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        rc = 124
    if out:
        out.close()
    if check and rc != 0:
        raise SystemExit(f'command failed ({rc}): {cmd}')
    return rc


def status(state, **kw):
    with LOCK:
        st = {'session': SESSION, 'state': state, 'time': time.strftime('%F %T'), 'jobs': JOBSTATE, **kw}
        json.dump(st, open(f'{OUT}/STATUS.json', 'w'), indent=1)
    log('STATUS', state, JOBSTATE)


def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True))
    return hits[0] if hits else None


def restore(archive, marker, dest):
    """Kaggle auto-extracts uploaded archives into files; accept either the archive or its extracted tree.
    marker: a path inside the archive (e.g. 'code/repro/kaggle/kernel.py'); its tree root is copied to dest."""
    a = find(archive)
    if a:
        tarfile.open(a).extractall(dest)
        return a
    m = find(marker)
    if not m:
        return None
    root = m[:-len(marker)].rstrip('/')
    for name in os.listdir(root):
        src = os.path.join(root, name)
        (shutil.copytree(src, os.path.join(dest, name), dirs_exist_ok=True) if os.path.isdir(src)
         else shutil.copy2(src, dest))
    return root


def sha_ok(path, name):
    want = [l.split()[0] for l in open(f'{PRJ}/weights/SHA256SUMS') if name in l][0]
    got = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    assert got == want, f'{name} checksum mismatch'


# ------------------------------------------------------------------------------------------- jobs
def job_spec(job):
    """-> dict(env, run_dir, epochs, fresh_cmd, resume_cmd, cwd)"""
    if job.startswith('p1_'):
        res = f'{OUT}/results/p1'
        cfg = {'p1_gbh': 'gbh_released.yaml', 'p1_yolov5s': 'yolov5s.yaml'}[job]
        py, cwd = f'{ENVS}/p1/bin/python', f'{PRJ}/code/gbh-yolov5'
        fresh = [py, 'train.py', '--cfg', f'{PRJ}/code/repro/p1/{cfg}', '--weights', f'{PRJ}/weights/yolov5s_v5.0.pt',
                 '--data', f'{PRJ}/datasets/pv_multi_defect_yolo/data.yaml', '--hyp', f'{PRJ}/code/repro/p1/hyp.gbh.yaml',
                 '--adam', '--epochs', '500', '--batch-size', '16', '--img-size', '960', '960', '--workers', '2',
                 '--device', '0', '--project', res, '--name', job, '--exist-ok']
        rd = f'{res}/{job}'
        return dict(run_dir=rd, epochs=500, cwd=cwd, fresh=fresh, resume=[py, 'train.py', '--resume', f'{rd}/weights/last.pt'],
                    done=lambda: sum(1 for _ in open(f'{rd}/results.txt')) if os.path.exists(f'{rd}/results.txt') else 0)
    res = f'{OUT}/results/p2_kaggle'
    py, data = f'{ENVS}/p2/bin/python', f'{PRJ}/datasets/eldds_yolo/data.yaml'
    rd = f'{res}/{job}'
    csv_rows = lambda: max(sum(1 for _ in open(f'{rd}/results.csv')) - 1, 0) if os.path.exists(f'{rd}/results.csv') else 0  # noqa
    if job.startswith('p2_v8s'):
        ep = [py, '-c', 'from ultralytics.yolo.cfg import entrypoint; entrypoint()']
        fresh = ep + ['detect', 'train', f'model={PRJ}/weights/yolov8s.pt', f'data={data}', 'optimizer=Adam', 'lr0=0.01',
                      'epochs=100', 'batch=32', 'imgsz=640', 'close_mosaic=10', 'seed=0', 'deterministic=True',
                      'workers=2', 'device=0', f'project={res}', f'name={job}', 'exist_ok=True']
        return dict(run_dir=rd, epochs=100, cwd=res, fresh=fresh, done=csv_rows,
                    resume=ep + ['detect', 'train', 'resume', f'model={rd}/weights/last.pt'])
    cfg = 'yolov5s-gam-asff.yaml' if 'improved' in job else 'yolov5s.yaml'
    fresh = [py, 'train.py', '--cfg', f'{PRJ}/code/repro/p2/{cfg}', '--weights', f'{PRJ}/weights/yolov5s_v7.0.pt',
             '--data', data, '--hyp', 'data/hyps/hyp.scratch-low.yaml', '--optimizer', 'Adam', '--epochs', '100',
             '--batch-size', '32', '--img', '640', '--seed', '0', '--workers', '2', '--device', '0',
             '--project', res, '--name', job, '--exist-ok']
    return dict(run_dir=rd, epochs=100, cwd=f'{PRJ}/code/yolov5-v7.0', fresh=fresh, done=csv_rows,
                resume=[py, 'train.py', '--resume', f'{rd}/weights/last.pt'])


def gpu_worker(gpu, jobs):
    env = {'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONIOENCODING': 'utf-8'}
    for job in jobs:
        s = job_spec(job)
        os.makedirs(os.path.dirname(s['run_dir']), exist_ok=True)
        if s['done']() >= s['epochs']:
            JOBSTATE[job] = {'state': 'finished', 'epochs': s['done'](), 'gpu': gpu}
            continue
        budget = SESSION_LIMIT_H - SAFETY_H - (time.time() - T0) / 3600
        if budget < 0.2:
            JOBSTATE[job] = {'state': 'incomplete', 'epochs': s['done'](), 'gpu': gpu}
            continue
        resume = os.path.exists(f"{s['run_dir']}/weights/last.pt")
        JOBSTATE[job] = {'state': 'training', 'epochs': s['done'](), 'gpu': gpu, 'resumed': resume}
        status('running')
        rc = sh(s['resume'] if resume else s['fresh'], cwd=s['cwd'], env=env, check=False,
                logfile=f'{OUT}/{job}_train.log', timeout=budget * 3600)
        n = s['done']()
        JOBSTATE[job] = {'state': 'finished' if n >= s['epochs'] else 'incomplete', 'epochs': n, 'gpu': gpu, 'rc': rc}
        status('running')


# ------------------------------------------------------------------------------------------- 1. bundle
os.makedirs(OUT, exist_ok=True)
os.makedirs(PRJ, exist_ok=True)
b = restore('bundle.tar.gz', 'code/repro/kaggle/kernel.py', PRJ)
assert b and os.path.exists(f'{PRJ}/code/repro/env/p1/uv.lock'), 'code bundle not attached'
log('code bundle from', b)
r = restore('resume.tar.gz', 'results/STATUS_RESUME_MARKER', OUT) or None
if not r:  # extracted resume dataset: any attached tree containing results/**/last.pt
    hits = sorted(glob.glob('/kaggle/input/**/results/**/weights/last.pt', recursive=True))
    if hits:
        src = hits[0][:hits[0].index('/results/') + len('/results')]
        shutil.copytree(src, f'{OUT}/results', dirs_exist_ok=True)
        r = src
if r:
    log('restored previous session outputs from', r)

if any(j.startswith('p2_') for g in PLAN for j in PLAN[g]):
    assert restore('eldds.tar.gz', 'datasets/raw/ELDDS1400C5/data.yaml', PRJ), 'ELDDS dataset not attached'
    log('ELDDS restored:', len(os.listdir(f'{PRJ}/datasets/raw/ELDDS1400C5/train/images')), 'train images')

# ------------------------------------------------------------------------------------------- 0. preflight
os.makedirs(OUT, exist_ok=True)
if sh('nvidia-smi', logfile=f'{OUT}/{SESSION}_machine.txt', check=False) != 0:
    status('no_gpu')
    raise SystemExit('no GPU in this session')
try:
    socket.create_connection(('pypi.org', 443), timeout=10).close()
except OSError as e:
    status('no_internet')
    raise SystemExit(f'no internet: {e}')
NGPU = int(subprocess.run('nvidia-smi -L | wc -l', shell=True, capture_output=True, text=True).stdout.strip() or 0)
if NGPU < 2:  # single GPU: run everything sequentially on GPU 0
    PLAN = {0: [j for g in sorted(PLAN) for j in PLAN[g]]}
log('GPUs', NGPU, 'plan', PLAN)
JOBS = [j for g in PLAN for j in PLAN[g]]
NEED_P1, NEED_P2 = any(j.startswith('p1_') for j in JOBS), any(j.startswith('p2_') for j in JOBS)

# ------------------------------------------------------------------------------------------- 2. envs (uv, locked)
sh(f'{sys.executable} -m pip install -q uv')
for p, need in (('p1', NEED_P1), ('p2', NEED_P2)):
    if need:
        sh(f'{sys.executable} -m uv sync --project {PRJ}/code/repro/env/{p} --frozen',
           env={'UV_PROJECT_ENVIRONMENT': f'{ENVS}/{p}'}, logfile=f'{OUT}/{SESSION}_setup.log')
        sh(f'{ENVS}/{p}/bin/python -c "import sys,torch;print(sys.version.split()[0],torch.__version__,torch.version.cuda,'
           f'torch.backends.cudnn.version(),[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"',
           logfile=f'{OUT}/{SESSION}_machine.txt')

# ------------------------------------------------------------------------------------------- 3. code, data, weights
os.makedirs(f'{PRJ}/weights', exist_ok=True)
if NEED_P1:
    sh(f'git clone -q https://github.com/CCNUZFW/GBH-YOLOv5 {PRJ}/code/gbh-yolov5')
    sh(f'git -C {PRJ}/code/gbh-yolov5 checkout -q {GBH_COMMIT}')
    sh(f'git -C {PRJ}/code/gbh-yolov5 apply --whitespace=nowarn {PRJ}/code/repro/p1/gbh_51ee042_fix.patch')
    sh(f'git clone -q https://github.com/CCNUZFW/PV-Multi-Defect {PRJ}/datasets/raw/PV-Multi-Defect')
    sh(f'git -C {PRJ}/datasets/raw/PV-Multi-Defect checkout -q {PV_COMMIT}')
    sh(f'curl -sSL -o {PRJ}/weights/yolov5s_v5.0.pt https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt')
    sha_ok(f'{PRJ}/weights/yolov5s_v5.0.pt', 'yolov5s_v5.0.pt')
    sh(f'{ENVS}/p1/bin/python -I {PRJ}/code/repro/voc2yolo_pv.py --src {PRJ}/datasets/raw/PV-Multi-Defect '
       f'--dst {PRJ}/datasets/pv_multi_defect_yolo --splits {PRJ}/datasets/splits/pv_multi_defect',
       logfile=f'{OUT}/{SESSION}_data.log')
    n = [len(open(f'{PRJ}/datasets/pv_multi_defect_yolo/{s}.txt').read().split()) for s in ('train', 'val')]
    assert n == [886, 222], n
if NEED_P2:
    sh(f'git clone -q --branch {V7_TAG} --depth 1 https://github.com/ultralytics/yolov5 {PRJ}/code/yolov5-v7.0')
    sh(f'git -C {PRJ}/code/yolov5-v7.0 apply --whitespace=nowarn {PRJ}/code/repro/p2/yolov5_v7.0_p2.patch')
    sh(f'{ENVS}/p2/bin/python -I {PRJ}/code/repro/prep_eldds.py --src {PRJ}/datasets/raw/ELDDS1400C5 '
       f'--dst {PRJ}/datasets/eldds_yolo --splits {PRJ}/datasets/splits/eldds1400c5', logfile=f'{OUT}/{SESSION}_data.log')
    n = [len(open(f'{PRJ}/datasets/eldds_yolo/{s}.txt').read().split()) for s in ('train', 'test')]
    assert n == [1186, 190], n
    for name, url in (('yolov5s_v7.0.pt', 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt'),
                      ('yolov8s.pt', 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt')):
        sh(f'curl -sSL -o {PRJ}/weights/{name} {url}')
        sha_ok(f'{PRJ}/weights/{name}', name)
    cfgdir = os.path.expanduser('~/.config/Ultralytics')  # py3.8 urllib can't follow the font URL's 308 redirect
    os.makedirs(cfgdir, exist_ok=True)
    for f in ('Arial.ttf', 'Arial.Unicode.ttf'):
        sh(f'curl -sSL -o {cfgdir}/{f} https://ultralytics.com/assets/{f}', check=False)
log('setup done')

# ------------------------------------------------------------------------------------------- 4. train (1 job/GPU)
threads = [threading.Thread(target=gpu_worker, args=(g, PLAN[g])) for g in sorted(PLAN)]
for t in threads:
    t.start()
for t in threads:
    t.join()
status('finished' if all(JOBSTATE.get(j, {}).get('state') == 'finished' for j in JOBS) else 'incomplete')
