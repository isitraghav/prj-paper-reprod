"""Build Kaggle notebooks (.ipynb) from kernel.py for manual runs, and optionally push them to the account.

Pushing always triggers one run; we push with GPU off, so that run stops at the preflight GPU check within
seconds and uses no GPU quota. Then, in the Kaggle UI: open the notebook -> Edit -> Settings: Accelerator
(GPU T4 x2 or P100), Internet On -> "Save Version" -> "Save & Run All (Commit)".

usage (repo root): uv run --no-project python code/repro/kaggle/make_notebooks.py [--push]
"""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results' / 'kaggle' / 'notebooks'
USER = 'tanishaggrawal69'
DS = {'s1': [f'{USER}/prj-paper-reprod-code'],
      's2': [f'{USER}/prj-paper-reprod-code', f'{USER}/prj-eldds1400c5-v1']}
TITLE = {'s1': 'repro-p1-manual', 's2': 'repro-p2-b32-manual'}
INTRO = {
    's1': 'P1 (GBH-YOLOv5 paper): **GBH-YOLOv5** on GPU 0 and **YOLOv5s** on GPU 1, at the paper\'s exact settings '
          '(Adam, lr 0.001, batch 16, 500 epochs, 960 px, COCO yolov5s.pt v5.0, torch 1.8.1+cu111).',
    's2': 'P2 (Mazen et al. 2023) at the paper\'s exact **batch 32**: **GAM+ASFF** then **YOLOv5s** on GPU 0, '
          '**YOLOv8s** on GPU 1 (Adam, lr 0.01, 100 epochs, 640 px, close_mosaic 10 for v8; '
          'py 3.8.8, torch 1.11.0+cu113, ultralytics 8.0.99).',
}


def notebook(session):
    code = (ROOT / 'code/repro/kaggle/kernel.py').read_text(encoding='utf8')
    code = code.replace("SESSION = '__SESSION__'", f"SESSION = {session!r}  # 's1' = P1, 's2' = P2 at batch 32")
    md = (f'# {TITLE[session]}\n\n{INTRO[session]}\n\n'
          '**Before running:** Settings → Accelerator **GPU T4 x2** (or P100: then jobs run one after another '
          'on the single GPU), **Internet On**. Inputs must include `prj-paper-reprod-code`'
          + (' and `prj-eldds1400c5-v1`' if session == 's2' else '') + '.\n\n'
          '**Run:** "Save Version" → "Save & Run All (Commit)" (runs in the background up to 12 h; outputs are saved). '
          'Interactive "Run All" also works but stops when the browser session ends.\n\n'
          '**Continue an unfinished run:** download this notebook\'s output, upload `results/` packed as '
          '`resume.tar.gz` as a dataset, add it as an input and run again: the script resumes from `last.pt`.\n\n'
          'Outputs: `/kaggle/working/results/...`, `*_train.log`, `STATUS.json`. Source: `code/repro/kaggle/kernel.py`.')
    cell = lambda t, src: {'cell_type': t, 'metadata': {}, 'source': src.splitlines(True),  # noqa: E731
                           **({'outputs': [], 'execution_count': None} if t == 'code' else {})}
    return {'cells': [cell('markdown', md), cell('code', code)], 'metadata': {
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
        'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--push', action='store_true')
    a = ap.parse_args()
    for s in ('s1', 's2'):
        d = OUT / s
        d.mkdir(parents=True, exist_ok=True)
        nb = d / f'{TITLE[s]}.ipynb'
        nb.write_text(json.dumps(notebook(s), indent=1), encoding='utf8')
        (d / 'kernel-metadata.json').write_text(json.dumps({
            'id': f'{USER}/{TITLE[s]}', 'title': TITLE[s], 'code_file': nb.name, 'language': 'python',
            'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': False, 'enable_internet': True,
            'dataset_sources': DS[s], 'competition_sources': [], 'kernel_sources': []}, indent=1))
        print('wrote', nb)
        if a.push:
            p = subprocess.run(['kaggle', 'kernels', 'push', '-p', str(d)], capture_output=True, text=True)
            print((p.stdout + p.stderr).strip().splitlines()[-1])


if __name__ == '__main__':
    main()
