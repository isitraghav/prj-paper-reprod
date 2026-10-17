"""Run trainings on Kaggle (T4 x2, one job per GPU) fully automatically via the Kaggle CLI.

Sessions (see PLAN in kernel.py):  s1 = P1 GBH + P1 YOLOv5s (batch 16, 960, 500 ep)
                                   s2 = P2 GAM+ASFF, YOLOv5s, YOLOv8s at the paper's exact batch 32
Per session: push notebook -> poll -> download outputs -> if unfinished, upload outputs as a private
"resume" dataset and push again (max --max-sessions). State: results/kaggle/state.json (restartable).

Needs `kaggle` CLI >= 2.x (uv tool install --python 3.12 kaggle==2.2.4) and credentials in ~/.kaggle
(outside the repo); phone-verified account.
usage (repo root): uv run --no-project python code/repro/kaggle/orchestrate.py [--sessions s1 s2] [--dry-run]
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / 'results' / 'kaggle'
STATE = WORK / 'state.json'
ACC = 'NvidiaTeslaT4'
CODE_DS, ELDDS_DS = 'prj-paper-reprod-code', 'prj-eldds1400c5-v1'
BUNDLE = ['code/repro', 'datasets/splits', 'weights/SHA256SUMS']
TERMINAL = ('COMPLETE', 'ERROR', 'CANCEL')
NEEDS_ELDDS = {'s2'}
SECRET = ('.env', 'kaggle.json', 'access_token', '.key')


def log(*a):
    s = f"[{time.strftime('%F %T')}] " + ' '.join(map(str, a))
    print(s, flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    with open(WORK / 'orchestrate.log', 'a', encoding='utf8') as f:
        f.write(s + '\n')


def kaggle(*args, check=True):
    p = subprocess.run(['kaggle', *args], capture_output=True, text=True, encoding='utf8', errors='replace')
    out = (p.stdout + p.stderr).strip()
    if check and p.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} failed: {out}")
    return out


def username():
    f = Path.home() / '.kaggle' / 'kaggle.json'
    if f.exists():
        return json.loads(f.read_text())['username']
    m = re.search(r'username:\s*(\S+)', kaggle('config', 'view', check=False))
    if m and m[1] != 'None':
        return m[1]
    raise SystemExit('no Kaggle credentials in ~/.kaggle/')


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(st):
    STATE.write_text(json.dumps(st, indent=1))


def make_tgz(path, items, base=ROOT):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, 'w:gz') as t:
        for rel in items:
            t.add(base / rel, arcname=rel,
                  filter=lambda ti: None if ('__pycache__' in ti.name or ti.name.endswith(SECRET)) else ti)
    names = tarfile.open(path).getnames()
    assert not any(n.endswith(SECRET) for n in names), f'secret in {path}'
    log(f'{path.name}: {len(names)} entries, {path.stat().st_size / 1e6:.1f} MB')
    return path


def wait_dataset(ref):
    for _ in range(160):
        s = kaggle('datasets', 'status', ref, check=False).lower()
        if 'ready' in s:
            return
        if 'error' in s:
            raise RuntimeError(f'dataset {ref}: {s}')
        time.sleep(15)
    raise RuntimeError(f'dataset {ref} not ready')


def upload_dataset(user, slug, files, msg):
    d = WORK / 'upload' / slug
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    for name, src in files.items():
        shutil.copy2(src, d / name)
    ref = f'{user}/{slug}'
    (d / 'dataset-metadata.json').write_text(json.dumps({'title': slug, 'id': ref, 'licenses': [{'name': 'CC0-1.0'}]}))
    if 'ready' in kaggle('datasets', 'status', ref, check=False).lower():
        log('versioning dataset', ref, kaggle('datasets', 'version', '-p', str(d), '-m', msg, '-q'))
    else:
        log('creating dataset', ref, kaggle('datasets', 'create', '-p', str(d), '-q'))
    time.sleep(20)
    wait_dataset(ref)
    return ref


def slug_of(session):
    return f'repro-{session}'


def kernel_folder(session, user, sources):
    kd = WORK / 'kernels' / session
    shutil.rmtree(kd, ignore_errors=True)
    kd.mkdir(parents=True)
    code = (ROOT / 'code/repro/kaggle/kernel.py').read_text(encoding='utf8').replace("'__SESSION__'", repr(session))
    (kd / 'kernel.py').write_text(code, encoding='utf8')
    (kd / 'kernel-metadata.json').write_text(json.dumps({
        'id': f'{user}/{slug_of(session)}', 'title': slug_of(session), 'code_file': 'kernel.py', 'language': 'python',
        'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_internet': True,
        'machine_shape': ACC, 'dataset_sources': sources, 'competition_sources': [], 'kernel_sources': []}, indent=1))
    return kd


def push(session, user, sources):
    kd = kernel_folder(session, user, sources)
    out = kaggle('kernels', 'push', '-p', str(kd), '--accelerator', ACC)
    log('pushed', session, '->', out.splitlines()[-1] if out else '')
    return f'{user}/{slug_of(session)}'


def kernel_status(ref):
    s = kaggle('kernels', 'status', ref, check=False)
    m = re.search(r'status "?(?:KernelWorkerStatus\.)?([A-Z_]+)', s)
    return m[1] if m else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', nargs='+', default=['s1', 's2'])
    ap.add_argument('--poll', type=int, default=300)
    ap.add_argument('--max-sessions', type=int, default=4)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    if a.dry_run:
        make_tgz(WORK / 'bundle.tar.gz', BUNDLE)
        for s in a.sessions:
            kernel_folder(s, 'USER', [])
        log('dry run ok')
        return

    user = username()
    st = load_state()
    log('kaggle user', user, '| sessions', a.sessions)
    if st.get('bundle_ver') != 2:
        st['code_ds'] = upload_dataset(user, CODE_DS, {'bundle.tar.gz': make_tgz(WORK / 'bundle.tar.gz', BUNDLE)},
                                       'bundle v2 (generic kernel)')
        st['bundle_ver'] = 2
        save_state(st)
    if NEEDS_ELDDS & set(a.sessions) and not st.get('eldds_ds'):
        st['eldds_ds'] = upload_dataset(user, ELDDS_DS, {'eldds.tar.gz': make_tgz(
            WORK / 'eldds.tar.gz', ['datasets/raw/ELDDS1400C5'])}, 'ELDDS1400C5 Roboflow v1 export')
        save_state(st)

    def sources(s, resume=None):
        src = [st['code_ds']] + ([st['eldds_ds']] if s in NEEDS_ELDDS else [])
        return src + ([resume] if resume else [])

    for s in a.sessions:
        ss = st.setdefault(s, {'n': 0, 'state': 'new'})
        if ss['state'] == 'new':
            ss['kernel'], ss['n'], ss['state'] = push(s, user, sources(s)), 1, 'running'
            save_state(st)

    while any(st[s]['state'] == 'running' for s in a.sessions):
        for s in a.sessions:
            ss = st[s]
            if ss['state'] != 'running':
                continue
            k = kernel_status(ss['kernel'])
            if ss.get('last') != k:
                log(s, 'session', ss['n'], 'status', k)
                ss['last'] = k
                save_state(st)
            if not k.startswith(TERMINAL):
                continue
            dl = WORK / s / f"session{ss['n']}"
            shutil.rmtree(dl, ignore_errors=True)
            dl.mkdir(parents=True)
            kaggle('kernels', 'output', ss['kernel'], '-p', str(dl), '-o', check=False)
            stj = dl / 'STATUS.json'
            info = json.loads(stj.read_text()) if stj.exists() else {'state': 'unknown'}
            log(s, 'session', ss['n'], 'ended', k, json.dumps(info))
            res = dl / 'results'
            if info.get('state') == 'finished':
                shutil.copytree(res, ROOT / 'results', dirs_exist_ok=True)
                ss['state'] = 'finished'
                log(s, 'FINISHED; results copied into results/')
            elif res.exists() and any(res.rglob('last.pt')) and ss['n'] < a.max_sessions:
                tgz = make_tgz(WORK / s / 'resume.tar.gz', ['results'], base=dl)
                ref = upload_dataset(user, f'{slug_of(s)}-resume', {'resume.tar.gz': tgz}, f"after session {ss['n']}")
                ss['kernel'] = push(s, user, sources(s, ref))
                ss['n'] += 1
                ss.pop('last', None)
                log(s, 'resumed as session', ss['n'])
            else:
                ss['state'] = 'failed'
                log(s, 'FAILED (no checkpoint, preflight failure or max sessions); see', dl)
            save_state(st)
        if any(st[s]['state'] == 'running' for s in a.sessions):
            time.sleep(a.poll)
    log('all done:', {s: st[s]['state'] for s in a.sessions})


if __name__ == '__main__':
    sys.exit(main())
