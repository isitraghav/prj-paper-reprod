"""Put the CUDA 11 / cuDNN 8 DLLs from the NVIDIA pip wheels (nvidia-*-cu11) on the DLL search path.
Import this before tensorflow on Windows."""
import os
import site
import sys
from pathlib import Path


def setup():
    if sys.platform != 'win32':
        return
    roots = [Path(p) / 'nvidia' for p in site.getsitepackages() + [site.getusersitepackages()]]
    dirs = [d for r in roots if r.is_dir() for d in r.glob('*/bin') if any(d.glob('*.dll'))]
    for d in dirs:
        os.add_dll_directory(str(d))
    os.environ['PATH'] = os.pathsep.join([str(d) for d in dirs] + [os.environ.get('PATH', '')])
    return dirs


setup()
