# Robust Automatic Detection of Mine-Like Contacts in Side-Scan Sonar Imagery

This repository contains the dataset, preprocessing pipelines, training configurations, model weights, and experimental results for robust automatic detection and classification of **MILCO** (Mine-Like Contacts) and **NOMBO** (Non-Mine-Like Contacts) in Side-Scan Sonar (SSS) imagery.

---

## 📁 Repository Structure

The repository is systematically organized into the following directories:

```text
├── 2010/                         # Raw side-scan sonar imagery & YOLO annotations (Survey Year 2010)
├── 2015/                         # Raw side-scan sonar imagery & YOLO annotations (Survey Year 2015)
├── 2017/                         # Raw side-scan sonar imagery & YOLO annotations (Survey Year 2017)
├── 2018/                         # Raw side-scan sonar imagery & YOLO annotations (Survey Year 2018)
├── 2021/                         # Raw side-scan sonar imagery & YOLO annotations (Survey Year 2021)
├── milco_dataset/                # Standardized, curated year-disjoint YOLO dataset
│   ├── images/                   # Organized images split by survey year (2010, 2015, 2017, 2018, 2021)
│   ├── labels/                   # Corresponding YOLO annotation .txt files by survey year
│   ├── plots/                    # Dataset distribution & bounding-box aspect ratio visualizations
│   ├── splits/                   # Manifests and text splits for training/validation/testing
│   ├── yolo_primary/             # YOLO configuration (data.yaml) and split manifest CSV
│   ├── dataset_bbox_analysis.csv # Bounding-box statistics across images
│   ├── dataset_image_index.csv   # Comprehensive index mapping images, survey years, and object counts
│   ├── dataset_year_summary.csv  # Annual summary of images, empty scenes, and contact distributions
│   ├── milco_year_disjoint.yaml  # Ultralytics dataset configuration for year-disjoint validation
│   ├── primary_experiment.yaml   # Primary benchmark split configuration
│   └── secondary_stress_test.yaml# Stress-test split configuration
├── Training/                     # YOLOv4 (Darknet) training scripts and model weights
│   ├── Real_time_object_classifier.ipynb # Notebook for Darknet YOLOv4 training and evaluation
│   ├── obj.names.txt             # Class names definition (MILCO, NOMBO)
│   ├── yolov4-custom.txt         # Darknet architecture configuration
│   ├── yolov4.conv.137           # Pretrained convolutional backbone weights (tracked via Git LFS)
│   └── yolov4-custom_5000.weights# Trained YOLOv4 model checkpoint (tracked via Git LFS)
├── runs/                         # Ultralytics YOLOv8 training runs and evaluation outputs
│   └── detect/                   # Training checkpoints, confusion matrices, loss curves, and PR curves
│       ├── yolov8s_primary_baseline/
│       ├── yolov8s_primary_baseline-2/
│       └── yolov8s_primary_baseline-3/
├── Untitled.ipynb                # End-to-end dataset curation, split generation, and YOLOv8 pipeline
├── yolov8s.pt                    # Pretrained YOLOv8-Small baseline weights (tracked via Git LFS)
├── .gitattributes                # Git LFS tracking rules for binary weights and archives
└── .gitignore                    # Ignored artifacts (caches, checkpoints, OS files)
```

---

## 📊 Dataset Overview

The dataset comprises side-scan sonar imagery collected across 5 survey years (2010, 2015, 2017, 2018, 2021), annotated for two classes:
- **Class 0 — MILCO** (Mine-Like Contact)
- **Class 1 — NOMBO** (Non-Mine-Like Contact)

### Annual Breakdown

| Survey Year | Total Images | Empty Images | MILCO Contacts | NOMBO Contacts | Avg Objects / Image |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **2010** | 345 | 317 | 22 | 12 | 0.10 |
| **2015** | 120 | 2 | 242 | 171 | 3.44 |
| **2017** | 93 | 74 | 28 | 2 | 0.32 |
| **2018** | 564 | 452 | 96 | 46 | 0.25 |
| **2021** | 48 | 21 | 49 | 0 | 1.02 |
| **TOTAL** | **1,170** | **866** | **437** | **231** | **0.57** |

---

## 🧪 Experimental Design: Year-Disjoint Validation

To prevent cross-survey data leakage caused by identical environmental and acoustic characteristics within the same survey year, the models are evaluated using a strict **year-disjoint protocol**:

- **Train Split**: Surveys from `2010`, `2015`, and `2017` (558 images, 292 MILCO, 185 NOMBO)
- **Validation Split**: Survey `2018` (564 images, 96 MILCO, 46 NOMBO)
- **Unseen Test Split**: Survey `2021` (48 images, 49 MILCO, 0 NOMBO)

---

## 📈 Benchmark Results: 100-Epoch YOLOv8 vs. YOLOv4

The models were benchmarked under identical task definitions (Class 0: MILCO, Class 1: NOMBO).

| Model | Evaluation Split | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **YOLOv4 Darknet (5,000 iter)** | Random Test Split | 51.0% | 80.0% | 75.5% | — | $\tau=0.01$; High False Alarm Rate (137 FP vs 140 TP) |
| **YOLOv8s (100 Epochs)** | Validation (84 img) | 74.1% | 51.2% | 57.6% | 30.9% | Baseline SOTA detector |
| **YOLOv8s (100 Epochs)** | Unseen 2018 Test (564 img) | 39.1% | 18.1% | 14.8% | 6.19% | Strict Year-Disjoint domain shift |
| **YOLO11s (100 Epochs)** | **Validation (84 img)** | **63.3%** | **47.6%** | **53.2%** | **30.2%** | **C3k2 & SPPF backbone architecture** |
| **YOLO11s (100 Epochs)** | **Unseen 2018 Test (564 img)** | **42.7%** | **18.5%** | **17.8%** | **7.18%** | **+20.0% mAP@0.50 gain over YOLOv8s under domain shift** |

### Direct Comparison on Unseen 2018 Test Survey (564 Images, 142 Targets):

| Metric | YOLOv8s (100 Epochs) | YOLO11s (100 Epochs) | Difference / Improvement |
| :--- | :---: | :---: | :---: |
| **mAP@0.50 (All Classes)** | 14.82% | **17.78%** | **+2.96% (+20.0% relative improvement)** |
| **mAP@0.50:0.95 (All Classes)** | 6.19% | **7.18%** | **+0.99% (+16.0% relative improvement)** |
| **MILCO mAP@0.50** | 17.03% | **25.10%** | **+8.07% (+47.4% relative improvement on mine targets!)** |
| **NOMBO mAP@0.50** | 12.60% | 10.45% | -2.15% |
| **True Positives (TP)** | 17 | **29** | **+70.6% (+12 True Positives detected)** |
| ↳ *MILCO TPs* | 12 | **24** | **+100% (Doubled detected mine targets!)** |
| ↳ *NOMBO TPs* | 5 | 5 | Equal (5) |
| **False Positives (FP)** | 29 | **28** | -1 (-3.4% fewer false alarms) |
| ↳ *FP on Empty Seabed* | 16 | **14** | -2 (-12.5% fewer empty seabed false alarms) |
| **False Negatives (FN)** | 125 | **113** | **-12 (-9.6% fewer missed contacts)** |
| **Operational Precision ($P$)** | 36.96% | **50.88%** | **+13.92% gain** |
| **Operational Recall ($R$)** | 11.97% | **20.42%** | **+8.45% gain** |
| **Operational F1-Score** | 0.1809 | **0.2915** | **+61.1% higher F1-score** |

---

## 🚀 Getting Started

### 1. Prerequisites & Git LFS
Model weights and binary archives are tracked using [Git Large File Storage (LFS)](https://git-lfs.github.com/). Ensure Git LFS is installed prior to cloning:

```bash
git lfs install
git clone https://github.com/CaptMJA380/Robust-Automatic-Detection-of-Mine-Like-Contacts-in-Side-Scan-Sonar-Imagery-Using-Sonar.git
cd Robust-Automatic-Detection-of-Mine-Like-Contacts-in-Side-Scan-Sonar-Imagery-Using-Sonar
git lfs pull
```

### 2. Dependencies
Install required Python packages:

```bash
pip install ultralytics torch torchvision pandas numpy matplotlib pyyaml
```

### 3. Training & Evaluation
To inspect or rerun the YOLOv8 pipeline, open `Untitled.ipynb` or launch training via Ultralytics CLI:

```bash
yolo task=detect mode=train model=yolov8s.pt data=milco_dataset/milco_year_disjoint.yaml epochs=100 imgsz=640 batch=16
```
