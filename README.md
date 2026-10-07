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
