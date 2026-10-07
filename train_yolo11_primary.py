import os
import sys
import time
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    print("=" * 80)
    print("STARTING ROBUST YOLO11s TRAINING ON MILCO/NOMBO YEAR-DISJOINT BENCHMARK")
    print("=" * 80)

    # 1. Hardware verification
    device = 0 if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[✓] Using GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    else:
        print("[!] CUDA not detected, running on CPU")

    data_yaml = Path("milco_dataset/yolo_primary/data.yaml").resolve()
    print(f"[✓] Dataset YAML: {data_yaml}")

    # 2. Model initialization
    weights_path = Path("yolo11s.pt").resolve()
    print(f"[✓] Loading base weights: {weights_path}")
    model = YOLO(str(weights_path))

    # 3. Training with exact same parameters as YOLOv8s:
    # - 100 epochs, patience=30
    # - batch=8, workers=0 (Windows CUDA stability)
    # - imgsz=640
    # - cos_lr=True, close_mosaic=10
    # - box=7.5, cls=1.5, dfl=1.5
    # - seed=42, deterministic=True
    print("\n[>>] Launching 100-Epoch YOLO11s Training...")
    start_time = time.time()
    train_results = model.train(
        data=str(data_yaml),
        epochs=100,
        imgsz=640,
        batch=8,
        workers=0,
        device=device,
        patience=30,
        cos_lr=True,
        close_mosaic=10,
        box=7.5,
        cls=1.5,
        dfl=1.5,
        seed=42,
        deterministic=True,
        name="yolo11s_primary_100epochs",
        exist_ok=True,
        verbose=True
    )
    train_duration = (time.time() - start_time) / 60.0
    print(f"\n[✓] Training completed in {train_duration:.2f} minutes!")

    # 4. Evaluation on Validation Set (84 images)
    print("\n" + "=" * 80)
    print("EVALUATING YOLO11s ON VALIDATION SPLIT (84 IMAGES)")
    print("=" * 80)
    val_metrics = model.val(
        data=str(data_yaml),
        split="val",
        name="yolo11s_val_eval"
    )

    # 5. Out-of-Distribution Evaluation on Unseen 2018 Test Split (564 images)
    print("\n" + "=" * 80)
    print("EVALUATING YOLO11s ON UNSEEN 2018 TEST SPLIT (564 IMAGES - YEAR-DISJOINT BENCHMARK)")
    print("=" * 80)
    test_metrics = model.val(
        data=str(data_yaml),
        split="test",
        name="yolo11s_test_2018_eval"
    )

    print("\n" + "=" * 80)
    print("SUMMARY OF FINAL YOLO11s METRICS")
    print("=" * 80)
    try:
        print(f"Validation mAP@0.50     : {val_metrics.box.map50:.4f}")
        print(f"Validation mAP@0.50:0.95: {val_metrics.box.map:.4f}")
        print(f"Unseen Test mAP@0.50    : {test_metrics.box.map50:.4f}")
        print(f"Unseen Test mAP@0.50:0.95: {test_metrics.box.map:.4f}")
    except Exception as e:
        print(f"Metrics printout error: {e}")

    print("\n[✓] All runs and weights saved under runs/detect/yolo11s_primary_100epochs")

if __name__ == "__main__":
    main()
