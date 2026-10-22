# Our code

Run from the repo root with the matching environment's Python and `-I`, e.g. `envs/p2/Scripts/python.exe -I ...`.
On Linux use `envs/<env>/bin/python`. Each script's docstring or header gives its exact usage.

## Data preparation (shared)
| Script | Does |
|---|---|
| `download_eldds.py` | Downloads ELDDS1400C5 from Roboflow (key from `ROBOFLOW_API_KEY` only) and writes `MANIFEST.txt` |
| `prep_eldds.py` | ELDDS Roboflow export → the paper's 6-class lists + `data.yaml` (P2) |
| `voc2yolo_pv.py` | PV-Multi-Defect VOC → YOLO format, using the frozen split (P1) |
| `draw_boxes.py` | Draws random labelled images into `results/sanity/` for an eyeball check |
| `mem_probe.py` | Finds the largest batch that fits in VRAM for a model config |
| `check_datasets.py` | Our three datasets vs each paper's stated counts → `results/dataset_check.md` |
| `summarize.py` | Collects the P2 eval logs into `results/summary.{csv,md}` |

## Run queues
| Script | Does |
|---|---|
| `run_queue_local.sh` | Laptop B queue: wait for P2 training → P2 eval + summary (→ P1 with `--with-p1`) |
| `setup_run_remaining.sh` | One-shot setup on the RTX 4050 machine: envs, code, data, then the remaining ablations |
| `run_p2_after_p1.sh` | Waits for P1 yolov5_2, then trains P2 v5_gam, then v5_asff, then runs the full P2 eval (pending) |

## Per paper
| Folder | Contents |
|---|---|
| `p1/` | GBH fork patch, model yamls (baseline, GBH released/paper, YOLOv5-1/-2 ablations), `hyp.gbh.yaml`, `run_p1.sh`, `model_check.py` (layer/parameter counts vs the paper), `leakage_check.py` (duplicate images, train/val copies, re-scored val) |
| `p2/` | yolov5 v7.0 patch (GAM, ASFF, DIoU-NMS), model yamls, `run_p2.sh`, `eval_p2.sh`, `v8_val.py` (YOLOv8 TTA), `gflops.py`, `make_yamls.py` |
| `p3/` | ELPV split, VGG-19 CNN training and evaluation, KAZE/VGG + VLAD + SVM training, the SVM bounded check |
| `env/` | Locked uv projects: `p1` (torch 1.8.1), `p2` (torch 1.11.0, ultralytics 8.0.99), `p3cnn` (TF 2.10.1), `p3svm` (OpenCV 3.4.18, sklearn 0.22) |
| `kaggle/` | Kaggle orchestration, tried and dropped (no GPU before phone verification); unused |

## `cross/`: cross-dataset comparison (COMPARISON.md)
Run in this order (commands in `COMPARISON.md` §10):

| Step | Script | Does |
|---|---|---|
| 1 | `run_dumps.sh` (`dump_v5.py`, `dump_v8.py`) | Every detector on every dataset, raw detections → `results/cross/preds/` |
| 2 | `make_tiles.py` | Tile grid + defective/functional labels for all three datasets → `results/cross/tiles.csv` |
| 3 | `cnn_tiles.py` | P3 CNN on every tile → `results/cross/preds/p3_cnn__tiles.csv` |
| 4 | `cross_eval.py` (`detmetrics.py`) | One evaluator for all models → `metrics.json`, `curves.json` |
| 5 | `report.py` | Figures 1–6, qualitative examples, `tables.md` |
| 6 | `run_sensitivity.sh`, `sensitivity.py` | Foreign input size and 12×12 tiles → `tables_sensitivity.md` |
| 7 | `class_map.py` | Class-mapped detection (crack / line / dark), Figs 7–8, `tables_classmap.md` |
| 8 | `paired_tests.py` | Paired bootstrap on model differences → `tables_paired.md` |

Predictions and tile CSVs (`preds*/`, `tiles*.csv`) are regenerated, not committed. `common.py` holds the dataset definitions, class names, class mapping and tiling shared by all of them.
