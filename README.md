# Reproduction and cross-dataset comparison of three PV defect-detection papers

This repo reproduces three papers on finding defects in solar (PV) module images. It then runs every trained model,
unchanged, on the other papers' datasets to see whether the models generalise. The results feed a review paper.

| ID | Paper | Task | Dataset | PDF |
|---|---|---|---|---|
| **P1** | Li et al. 2023, *GBH-YOLOv5* (Electronics 12, 561) | box detection, 5 classes | PV-Multi-Defect (600×600 module crops) | `papers/yoloGBH-paper1.pdf` |
| **P2** | Mazen et al. 2023, *Deep Learning for Automatic Defect Detection in PV Modules Using EL Images* (IEEE Access 11) | box detection, 6 classes | ELDDS1400C5 (whole EL panels) | `papers/Deep_Learning_for_Automatic_Defect_Detection_in_PV_Modules_Using_Electroluminescence_Images.pdf` |
| **P3** | Deitsch et al. 2019, *Automatic classification of defective PV module cells in EL images* (Solar Energy 185) | cell classification, defective vs functional | ELPV (300×300 single cells) | `papers/el-auto-cnn-svg.pdf` |

## Results in one paragraph

**Reproduction (home dataset).** P2 and the P3 CNN reproduce closely; the P3 SVM falls short (69.7–79.7 vs 82.4 % accuracy). P1's baseline does better than the paper reports, but
its GBH model does not reach the paper's 97.8 mAP; it is in fact below its own baseline. Details: `models/README.md`.

**Cross-dataset (the comparison).** No model generalises. On a foreign dataset every detector drops from mAP@.5
0.46–0.82 to 0.004–0.053. Matching the class names by visual similarity only lifts that to 0.05–0.13. The best
defective-vs-functional AUC any model reaches on foreign data is 0.61, against 0.88–0.98 at home, and 7 of the 12
cross cells are at chance. The one defect that carries over is a dark, unpowered cell column. The papers'
architecture improvements make little or no difference, and none of them changes transfer. Full report with
figures: **`COMPARISON.md`**.

## Documents

| File | What it is |
|---|---|
| **`REPORT.md`** | **The full report for the review: motivation, objectives, both replications, home and cross-dataset results, conclusion** |
| `COMPARISON.md` | The cross-dataset comparison: protocol, results, figures, discussion and limitations for the review |
| `CLAIMS.md` | Every checkable claim of the three papers, marked confirmed / partly / not confirmed / not tested, plus internal inconsistencies |
| `init.md` | Runbook: settings taken from each paper, environment setup, deviations from the papers (§11, P3-…), progress log (§13) |
| `CHECKLIST.md` | Status tracker per paper (environments, data, training, evaluation), the comparison (§9) and what is left (§10) |
| `models/README.md` | The final trained models (Git LFS), their home scores against the papers, and how to load them |
| `code/repro/README.md` | What each of our scripts does |
| `4070_JOBS.md` | Step-by-step job sheet used to train P1 on the RTX 4070 laptop (done; kept as a record) |

## Layout

```
papers/            the three PDFs
code/repro/        our code (see code/repro/README.md)
  p1/ p2/ p3/      per-paper configs, patches, training and evaluation scripts
  cross/           the cross-dataset comparison
  replication/     re-scoring of the earlier R1 run (REPORT.md §5.1)
  env/             locked uv projects for each environment (p1, p2, p3cnn, p3svm)
  kaggle/          Kaggle tooling (tried, dropped; unused)
code/gbh-yolov5/   P1 authors' code @51ee042           (cloned, not committed)
code/yolov5-v7.0/  ultralytics/yolov5 v7.0 for P2       (cloned, not committed)
datasets/splits/   frozen train/val/test lists (committed); the images are re-downloaded (init.md §5)
weights/           SHA256 of the COCO init weights (the .pt files are not committed)
models/            final trained models, Git LFS
results/
  p1/ p2/ p3/      training curves, logs and evaluation output per paper
  cross/           comparison numbers (*.json), tables (tables*.md), figures (figs/)
  replication_r1/  the earlier RTX 3050 run (R1) re-scored with our evaluators
  sanity/          box drawings used to check the label conversion
  summary.*        P2 evaluation summary (paper vs ours)
envs/              Python environments (built per machine, not committed)
```

## Reproduce

Environments, code and data: `init.md` §4–5. Training and evaluation per paper: `CHECKLIST.md`. The comparison:
`COMPARISON.md` §10. Run Python with `-I`. The Roboflow key (P2 data) is read only from the `ROBOFLOW_API_KEY`
environment variable and is never written to disk.

## Still open

Two ablation runs are not needed for the comparison: P1 YOLOv5-2, which stopped at epoch 54, and P2 GAM-only and
ASFF-only. See `CHECKLIST.md` §10.
