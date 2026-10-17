"""Download ELDDS1400C5 (P2 dataset) from Roboflow in yolov5 format.

The key comes from $ROBOFLOW_API_KEY or, failing that, from the gitignored
<repo>/.env (ROBOFLOW_API_KEY=...). It is never printed or written elsewhere.

Run in a throwaway env so roboflow's opencv-headless pin doesn't touch envs/p2:
  uv run --no-project --python 3.10 --with roboflow python -I code/repro/download_eldds.py \
      --out datasets/raw/ELDDS1400C5 [--list] [--version N]
"""
import argparse
import os
from pathlib import Path

WORKSPACE = 'fma04-fayoum-edu-eg'
PROJECT = 'eldds1400c5-dataset'


def api_key(repo):
    k = os.environ.get('ROBOFLOW_API_KEY')
    env = repo / '.env'
    if not k and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith('ROBOFLOW_API_KEY='):
                k = line.split('=', 1)[1].strip()
    if not k:
        raise SystemExit('ROBOFLOW_API_KEY not set (env var or .env)')
    return k


def count(d):
    return {s: len(list((d / s / 'images').glob('*'))) for s in ('train', 'valid', 'test') if (d / s / 'images').is_dir()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--list', action='store_true', help='only list versions')
    ap.add_argument('--version', type=int, help='version id (default: newest without augmentation)')
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]

    from roboflow import Roboflow
    proj = Roboflow(api_key=api_key(repo)).workspace(WORKSPACE).project(PROJECT)

    infos = []
    for v in proj.versions():
        aug = getattr(v, 'augmentation', None) or {}
        pre = getattr(v, 'preprocessing', None) or {}
        vid = int(str(v.version).rstrip('/').split('/')[-1])
        infos.append((vid, v, aug, pre))
        print(f'version {vid}: name={getattr(v, "name", "")!r} splits={getattr(v, "splits", "")} '
              f'augmentation={aug} preprocessing={pre}')
    if a.list:
        return

    if a.version is not None:
        pick = next(i for i in infos if i[0] == a.version)
    else:
        clean = [i for i in infos if not i[2]]
        if not clean:
            raise SystemExit('every version has augmentation; pick one with --version')
        pick = max(clean, key=lambda i: i[0])
    vid, v, aug, pre = pick

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    v.download('yolov5', location=str(out), overwrite=True)

    import yaml
    data = yaml.safe_load((out / 'data.yaml').read_text())
    counts = count(out)
    lines = [f'workspace: {WORKSPACE}', f'project: {PROJECT}', f'version: {vid}',
             f'augmentation: {aug}', f'preprocessing: {pre}', f'counts: {counts}',
             f'total: {sum(counts.values())}', f'nc: {data.get("nc")}', f'names: {data.get("names")}']
    (out / 'MANIFEST.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
