"""YOLOv8 validation with optional TTA for ultralytics 8.0.99.

8.0.99's validator calls model(img) without `augment`, so `yolo val augment=True` silently runs plain
inference. With --tta this wrapper forces AutoBackend to call DetectionModel._forward_augment
(horizontal flip + scales 1/0.83/0.67, as described in P2). Everything else is stock `yolo detect val`.

usage: python v8_val.py [--tta] model=best.pt data=data.yaml split=test imgsz=640 [project=.. name=..]
"""
import sys

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.cfg import entrypoint


def main():
    if '--tta' in sys.argv:
        sys.argv.remove('--tta')
        _fwd = AutoBackend.forward

        def forward(self, im, augment=False, visualize=False):
            return _fwd(self, im, augment=True, visualize=visualize)

        AutoBackend.forward = forward
        sys.argv.append('augment=True')  # recorded in args.yaml; behaviour comes from the patch above
        print('v8_val: TTA forced on (AutoBackend.forward augment=True)')
    sys.argv[1:1] = ['detect', 'val']
    entrypoint()


if __name__ == '__main__':  # required on Windows: dataloader workers re-import this file
    main()
