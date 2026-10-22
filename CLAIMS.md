# What each paper claims, and whether our reproduction confirms it

Claim-by-claim check of the three papers in `papers/` against our reproduction. Paper quotes and numbers are from
the PDFs. Our numbers are from `models/README.md`, `CHECKLIST.md` §6–8, `init.md` (P3 section), `results/` and
`COMPARISON.md`. Date: 2026-10-22.

| Mark | Meaning |
|---|---|
| ✅ | Confirmed |
| ◐ | Partly confirmed |
| ❌ | Not confirmed |
| — | Not tested; the reason is given |

**Caveats that apply to every row:**
- **One training run per model**, except the P3 CNN, which has three seeds. The papers report ±95 % CIs.
- **Batch size:** our physical batch is smaller than the papers' (P1 GBH, P2). Accumulation keeps 64 images per
  optimiser step.
- **P1 split:** the train/val split is re-created, because the authors' split was never published.
- **P2 architecture:** GAM/ASFF placement is inferred from the paper's GFLOPs, because no code was released.

## Summary

| Paper | Headline claim | Verdict |
|---|---|---|
| P1 | GBH-YOLOv5 reaches 97.8 mAP@.5, +19.7 over YOLOv5s | ❌ 82.1, **4.4 below** our YOLOv5s (86.5). Significant in a paired test: −0.033 [−0.061, −0.007] |
| P2 | Improved YOLOv5 reaches 76.3 mAP@.5, +2.5 over YOLOv5s | ❌ 0.757, +0.003 over YOLOv5s (not significant: [−0.008, +0.015]). The models' absolute levels do reproduce (within ±0.016) |
| P3 | CNN 88.42 % accuracy beats SVM 82.44 % | ◐ The CNN > SVM ranking and the ≈ 6-point gap reproduce. Both are lower: CNN 86.5 %, SVM 79.7 % (69.7 % with the paper's stated defaults) |

## P1: Li et al. 2023, GBH-YOLOv5 (Electronics 12, 561)

| # | Paper claims | Ours | Verdict |
|---|---|---|---|
| 1.1 | Dataset: 1108 images, 600 × 600, 886 train / 222 val | 1108 images (2 without labels are defect-free backgrounds), 886 / 222 | ✅ |
| 1.2 | 4235 defect boxes; hot_spot 49.09 %, scratch 36.62 %, black_border 6.02 %, no_electricity 4.28 %, broken 3.99 % (Fig. 5) | Released annotations: **3981** boxes; 52.22 / 34.34 / 6.43 / 4.55 / **2.46 %** | ❌ The released data has 254 fewer boxes (−6 %) and 42 % fewer `broken` boxes (98 vs ≈ 169) |
| 1.3 | GBH-YOLOv5 mAP@.5 97.8, P 96.4, R 93.3 (Table 4) | 82.1, P 81.1, R 81.2 (authors' code and released config, 500 epochs at 960 px) | ❌ |
| 1.4 | YOLOv5s baseline mAP@.5 78.1 (Table 4) | **86.5** | ❌ Our baseline is 8.4 higher |
| 1.5 | Adding BottleneckCSP (YOLOv5-1) gives +16.1 mAP, to 94.2 (Table 4) | YOLOv5-1, layer/param count identical to the paper: 85.2, i.e. **−1.3** vs our YOLOv5s | ❌ |
| 1.6 | Extra tiny-target head (YOLOv5-2): 97.1, +2.9 (Table 4) | Run stopped at epoch 54 / 500 (Git Bash fork crash) | — Pending (`CHECKLIST.md` §10) |
| 1.7 | GBH ≥ 97 AP on every class, scratch 97.4; the tiny-target head "alleviates tiny defect misses" (Table 5) | GBH 63.4–92.3 per class. Scratch: GBH 63.4 vs YOLOv5s 65.5 | ❌ The tiny-target design does not improve the tiny class |
| 1.8 | Layers / params: YOLOv5s 224 / 7.06 M, YOLOv5-1 228 / 7.15 M (Table 6) | 224 / 7.065 M and 228 / 7.153 M | ✅ Exact |
| 1.9 | Layers / params: GBH 270 / 7.24 M, YOLOv5-2 290 / 7.72 M (Table 6) | Released GBH config: 290 / 7.04 M. None of 32 Conv/Ghost × C3/BCSP × 3/4-head variants gives 270 or 7.72 M | ❌ The published GBH architecture cannot be rebuilt from the paper or the repo |
| 1.10 | GhostConv makes GBH faster than YOLOv5-2: −0.108 s/img (Table 6) | GBH 16.5 ms vs YOLOv5s 11.7 ms; YOLOv5-2 not trained | — Not tested. The paper's 0.48–0.70 s per image on an RTX 3090 is ≈ 30× our laptop's times, so it cannot be pure inference |
| 1.11 | Beats five other methods (YOLOv3, Faster R-CNN, SVM, Mask R-CNN, SSD) by "at least 27.8 %" (Table 3) | Not retrained | — Not tested. From the paper's own Table 3 the margin is 97.8 − 69.3 = **28.5** points (41 % relative), not 27.8 |
| 1.12 | Compress-and-crop preprocessing lifts mAP from 0.473 to 0.781 (Fig. 7) | Only the cropped 600 × 600 images are released, not the 307 raw 5800 × 3504 photos | — Not testable |
| 1.13 | Implied: works beyond its dataset (generalisation is attributed to deep learning, §1) | Cross-dataset class-agnostic mAP@.5 0.035–0.053; tile AUC 0.58–0.61 (`COMPARISON.md`) | ❌ The paper's own conclusion already warns about errors "in a natural production environment" |

**Internal inconsistencies in P1** (arithmetic on the paper's own numbers):
- **Table 5 vs Table 4.** The per-class APs of Table 5 should average to the mAP of Table 4, but they don't:
  - YOLOv5s: per-class mean **81.8**, Table 4 says 78.1;
  - YOLOv5-1: per-class mean **97.3**, Table 4 says 94.2;
  - GBH: 97.9 vs 97.8, which matches.

  Fig. 7 has the same problem. Its "after" values average 0.817 but it states 0.781, while its "before" values do average to the stated 0.473. Our own per-class values average exactly to our mAP (86.48 → 86.5).
- **P and R swapped.** GBH's precision and recall appear swapped between Table 2 (epoch 500: R 96.4, P 93.3) and Table 4 (P 96.4, R 93.3).
- **Taken together:** the reported baseline (78.1) and YOLOv5-1 (94.2) are lower than the paper's own per-class numbers imply, which inflates the ablation gains.

## P2: Mazen et al. 2023, Improved YOLOv5 / YOLOv8 on ELDDS1400C5 (IEEE Access 11)

Test split, 190 images, 640 px. Our numbers come from each codebase's own validator (`results/summary.md`).

| # | Paper claims | Ours | Verdict |
|---|---|---|---|
| 2.1 | Dataset: 1187 train / 190 test, 6 classes (Fig. 2) | 1186 / 190 (2 images whose only labels are classes outside the paper's six were excluded), 6 classes | ✅ |
| 2.2 | YOLOv5s mAP@.5 0.739, mAP@.5:.95 0.502 | 0.754, 0.516 | ✅ (+0.015) |
| 2.3 | YOLOv8s mAP@.5 0.765 | 0.764 | ✅ |
| 2.4 | Improved (GAM + ASFF + DIoU-NMS) 0.763, i.e. **+2.5 points over YOLOv5s** (abstract, conclusion) | 0.757, **+0.003**; paired bootstrap [−0.008, +0.015] | ◐ The absolute level is close (−0.006). ❌ The improvement does not reproduce |
| 2.5 | Improved has higher P and R than YOLOv5s (0.735 / 0.760 vs 0.726 / 0.723) | 0.754 / 0.727 vs 0.748 / 0.707 | ◐ Same direction, smaller gaps |
| 2.6 | "Using TTA, the mAP@0.5 of all models was improved" (Table 2) | YOLOv5s +0.012 (0.766 = paper), Improved +0.001, YOLOv8s **−0.011** | ◐ Only YOLOv5s gains as reported |
| 2.7 | With TTA, Improved (0.777) beats YOLOv8 (0.775) | 0.758 vs 0.753 | ◐ Same order, but both are lower than the paper and 0.005 apart. At TTA 832 px, plain YOLOv5s is best (0.774) |
| 2.8 | With TTA, Improved beats YOLOv8 on Crack and MicroCrack, which "validate[s] the significant effect of the GAM attention module" | Crack 0.781 vs 0.754, MicroCrack 0.325 vs 0.299 | ✅ for the comparison. ❌ for the explanation: plain YOLOv5s with TTA, which has no GAM, scores higher still (0.784 and 0.359) |
| 2.9 | F1 max: TTA-Improved 0.74, TTA-YOLOv8 0.76 (Fig. 12) | 0.74 (at conf 0.460) and 0.75 (at 0.411) | ✅ |
| 2.10 | GFLOPs: YOLOv5s 16, Improved 30.0, YOLOv8s 28.7; GAM-only 21.4, ASFF-only 25.8 | 16.0, 30.0, 28.7 | ✅ YOLOv5s and YOLOv8s match independently. Improved matches by construction, because we placed GAM/ASFF to fit the paper's GFLOPs |
| 2.11 | YOLOv8 is faster: ≈ 6 ms vs ≈ 13 ms per image for the YOLOv5 variants | Batch 1 inference: YOLOv8s 12.6 ms, YOLOv5s 12.4 ms, Improved 24.0 ms | ◐ YOLOv8s is faster than Improved but not faster than YOLOv5s. Timing depends on hardware (paper: P100) |
| 2.12 | Beats the YOLOv5l of a previous study on ELDDS1400C5 (Table 3) | Not run | — Not tested |
| 2.13 | Implied: "great promise for … automated defect detection and quality control in the PV industry" | On P1's images: class-agnostic mAP@.5 0.004–0.015, tile AUC at chance. The only transfer is "dark area" (AP 0.13–0.35 on 36 boxes) | ❌ Outside ELDDS1400C5 |
| 2.14 | Ablations GAM-only, ASFF-only, DIoU-only (Tables 1–2) | Not trained | — Pending (`CHECKLIST.md` §10) |

**Internal inconsistencies in P2:**
- **Class count:** the text says 5 classes, but Fig. 2 and the dataset have 6.
- **Image count:** the conclusion says the dataset "comprises 2,426 high-resolution solar cells images", but 1187 + 190 images are used.

## P3: Deitsch et al. 2019, CNN vs SVM on ELPV (Solar Energy 185)

Test split, 656 cells. Accuracy and F1 are weighted by label confidence (Table 2 weights).

| # | Paper claims | Ours | Verdict |
|---|---|---|---|
| 3.1 | Split: 1968 train / 656 test cells, per type and label (Table 3) | Identical | ✅ |
| 3.2 | CNN accuracy 88.42 %, F1 88.39 % | 86.49 / 84.20 (seed 0). Three seeds: 85.78 ± 0.76 / 83.38 ± 0.77 | ◐ Accuracy is 1.9–2.6 below the paper. F1 is 4–5 below, consistently across seeds |
| 3.3 | SVM (KAZE/VGG + VLAD + linear SVM) accuracy 82.44 %, F1 82.52 %, AUC 88.51 % | Paper's stated defaults: 69.70 / 66.32 / 74.88. With the unstated details picked by training-CV: 79.65 / 77.09 / 84.08 | ◐ Only after filling in details the paper doesn't state. The gap comes from keypoint density |
| 3.4 | CNN is more accurate than the SVM | 86.5 vs 79.7 (or 69.7) | ✅ |
| 3.5 | The two differ "in total by only about 6 %" | 6.8 points (better-specified SVM) | ✅ (16.8 with the paper's defaults) |
| 3.6 | Mono cells: CNN and SVM "on par". Poly cells: CNN clearly better (≈ 11 % AUC, ≈ 6 % accuracy) | Mono gap: AUC_w +4.8, acc_w +5.6. Poly gap: AUC_w +6.8, acc_w +7.7 | ◐ The gap is larger on poly, as claimed, but mono is not on par |
| 3.7 | Performance grows roughly logarithmically with training-set size (Fig. 10) | Not run | — Not tested |
| 3.8 | Confidence weighting helps; keypoints beat dense grids; masking rarely helps (§4.2–4.3) | Only the selected configuration was built | — Not tested |
| 3.9 | Runtime: SVM training ≈ 30 min, CNN ≈ 5 h; CNN inference ≈ 20 s on GPU vs SVM ≈ 8 min | CNN 100 + 100 epochs ≈ 3.4 h on an RTX 3050 Laptop | — Not comparable (different hardware) |
| 3.10 | Limitations stated: each cell is judged on its own; use outside a manufacturing setting has challenges | CNN on P1's tiles: AUC 0.50 (chance); on ELDDS tiles: 0.59 | ✅ Consistent. P3 is the one paper whose stated caution matches our cross-dataset result |

## What this means for the review

1. **Where a paper only *uses* standard models, its numbers reproduce:**
   - P1's YOLOv5s comes out even higher than reported;
   - P2's YOLOv5s and YOLOv8s are within ±0.016;
   - P3's CNN is within 2–3 accuracy points.
2. **Where a paper claims its *own improvement*, the improvement does not reproduce:**
   - P1's GBH is below its baseline;
   - P1's BottleneckCSP gain is −1.3, not +16.1;
   - P2's Improved model is +0.003, not +2.5, and its small-defect advantage is shared by plain YOLOv5s.

   P3 makes no architecture claim beyond CNN > SVM, and that one holds.
3. **P1's tables contradict each other.** Its reported baseline and YOLOv5-1 numbers are lower than its own per-class
   values imply, which inflates the ablation gains.
4. **No paper tests on a second dataset, and none of the models transfers** (`COMPARISON.md`). Only P3 warns about
   this explicitly.
