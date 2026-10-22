# Requests to the authors (drafts, not sent)

These are ready-to-send drafts. Each one asks only for what closes a gap that public files cannot close
(`CLAIMS.md`, `REPORT.md` §5.4). Add your name and affiliation before sending. The P1 paper's data statement says
data "will be made available on reasonable request".

---

## P1: Li, Wang, Zhang 2023, GBH-YOLOv5 (Electronics 12, 561)

**To:** zfwang@ccnu.edu.cn
**Subject:** Reproducing GBH-YOLOv5 (Electronics 2023, 12, 561): request for split lists, annotations and model config

Dear Prof. Wang,

We are reproducing GBH-YOLOv5 for a review of PV defect-detection methods. We used your repository
(CCNUZFW/GBH-YOLOv5 @ 51ee042), the released PV-Multi-Defect dataset, and the paper's settings: Adam, lr 0.001,
batch 16, 500 epochs, 960 × 960 input. Our YOLOv5s baseline reaches mAP@.5 86.5. Our GBH-YOLOv5 reaches 82.1 with the
released `models/yolov5s_pv.yaml`, against 97.8 in Table 4. Three differences from the paper might explain this, and
we would be grateful for the following:

1. **The train/val split** (the 886 / 222 image lists). The split is random and was not published.
2. **The annotation version used in the paper.** Fig. 5 reports 4,235 boxes. The released `Annotations/` folder has
   3,981: 184 scratch and 71 broken boxes fewer, while the other three classes match exactly.
3. **The exact model configuration or trained weights of GBH-YOLOv5** (and of YOLOv5-2). Table 6 reports 270 layers
   and 7.24 M parameters; the released yaml gives 290 layers and 7.04 M, and we could not find a configuration with
   270 layers.
4. If possible, the 307 raw images used for the compress-and-crop preprocessing (Fig. 7).

We will acknowledge your help and report the results with your files separately from ours.

Kind regards,
[name, affiliation]

---

## P2: Mazen et al. 2023, Deep Learning for Automatic Defect Detection in PV Modules Using EL Images (IEEE Access 11)

**To:** fma04@fayoum.edu.eg
**Subject:** Reproducing your improved YOLOv5 on ELDDS1400C5 (IEEE Access 2023): request for GAM/ASFF code and split

Dear Dr. Mazen,

We are reproducing your study for a review of PV defect-detection methods, using the ELDDS1400C5 dataset from your
Roboflow workspace and the settings in your paper: Adam, lr 0.01, 100 epochs, 640 × 640, batch 32 with
accumulation. Our YOLOv5s (0.754) and YOLOv8s (0.764) agree closely with your Table 2. Our GAM + ASFF + DIoU-NMS model
reaches 0.757, i.e. +0.003 over YOLOv5s rather than +0.023. Because no code was released, we had to infer the module
placement from your GFLOPs. We would be grateful for:

1. **The model code or yaml for GAM and ASFF**: the layers after which GAM is inserted, the ASFF variant, and whether
   DIoU is used for NMS only or also in the loss.
2. **The train/test split.** The text says 1,187 / 190 images, while Fig. 2 shows 1,240 / 138 and the Roboflow export
   has 1,188 / 190.
3. **The trained weights**, if you can share them, and the number of training runs behind Table 2.

Kind regards,
[name, affiliation]

---

## P3: Deitsch et al. 2019, Automatic classification of defective PV module cells in EL images (Solar Energy 185)

**To:** sergiu.deitsch@fau.de
**Subject:** Reproducing the ELPV CNN and SVM results (Solar Energy 2019): request for the train/test split

Dear Dr. Deitsch,

We are reproducing your CNN and KAZE/VGG SVM results on the ELPV dataset for a review of PV inspection methods.
With your stated settings our CNN reaches 86.5 % weighted accuracy (three seeds 85.8 ± 0.8; paper 88.42 %). Our
SVM reaches 69.7 % with OpenCV's default KAZE threshold and 79.7 % with denser keypoints (paper 82.44 %). We would be
grateful for:

1. **The 1,968 / 656 train/test split** (cell file names), which was not published. We used a stratified split with
   your Table 3 counts.
2. **The VLAD codebook size, k-means subset size and PCA output dimension** of the SVM pipeline, and the KAZE
   threshold, if it differed from the OpenCV 3.3.1 default.
3. **The CNN's output activation** (linear or sigmoid), and whether the reported F1 is macro- or support-weighted.

Kind regards,
[name, affiliation]
