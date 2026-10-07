import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    unionArea = box1Area + box2Area - interArea
    if unionArea <= 0:
        return 0.0
    return interArea / unionArea

def main():
    print("=" * 80)
    print("SWEEPING CONFIDENCE THRESHOLDS FOR YOLO11s ON 2018 TEST SET (564 IMAGES)")
    print("=" * 80)

    model_path = Path("runs/detect/yolo11s_primary_100epochs/weights/best.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing weights: {model_path}")

    model = YOLO(str(model_path))
    test_img_dir = Path("milco_dataset/yolo_primary/images/test")
    test_lbl_dir = Path("milco_dataset/yolo_primary/labels/test")

    img_paths = sorted(list(test_img_dir.glob("*.jpg")))
    print(f"Total test images: {len(img_paths)}")

    # Pre-parse all Ground Truth labels and dimensions once
    dataset_gt = {}
    total_gt_count = 0
    for img_p in img_paths:
        stem = img_p.stem
        lbl_p = test_lbl_dir / f"{stem}.txt"
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]
        
        gts = []
        if lbl_p.exists():
            for line in lbl_p.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x1 = (xc - w / 2.0) * w_img
                    y1 = (yc - h / 2.0) * h_img
                    x2 = (xc + w / 2.0) * w_img
                    y2 = (yc + h / 2.0) * h_img
                    gts.append({
                        "cls": cls_id,
                        "box": [x1, y1, x2, y2]
                    })
        dataset_gt[stem] = {"gts": gts, "w": w_img, "h": h_img}
        total_gt_count += len(gts)

    print(f"Total Ground Truth contacts across dataset: {total_gt_count}")

    # Run inference ONCE at the lowest threshold (conf=0.05) to cache all candidate boxes and confidences
    print("\n[>>] Running batch inference across test images at low threshold (conf=0.05)...")
    dataset_preds = {}
    for img_p in img_paths:
        stem = img_p.stem
        results = model.predict(source=str(img_p), conf=0.05, imgsz=640, device=0, verbose=False)[0]
        preds = []
        for b in results.boxes:
            c = int(b.cls[0].item())
            cf = float(b.conf[0].item())
            bx = b.xyxy[0].tolist()
            preds.append({
                "cls": c,
                "conf": cf,
                "box": bx
            })
        dataset_preds[stem] = preds

    print("[✓] Raw predictions cached. Evaluating requested confidence thresholds...")

    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    IOU_THRESH = 0.50

    results_table = []

    for conf in thresholds:
        tp_total = 0
        fp_total = 0
        fn_total = 0
        tp_milco = 0
        tp_nombo = 0
        fp_milco = 0
        fp_nombo = 0
        fn_milco = 0
        fn_nombo = 0
        fp_empty_seabed = 0

        for stem, data in dataset_gt.items():
            gts = [{"cls": g["cls"], "box": g["box"], "matched": False} for g in data["gts"]]
            # Filter cached predictions for current threshold
            filtered_preds = [p for p in dataset_preds[stem] if p["conf"] >= conf]

            is_empty_seabed = (len(gts) == 0)

            # Match predictions to GT (IoU >= 0.50, same class)
            for p in filtered_preds:
                best_iou = 0.0
                best_gt_idx = -1
                for g_idx, g in enumerate(gts):
                    if not g["matched"] and g["cls"] == p["cls"]:
                        iou = compute_iou(p["box"], g["box"])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = g_idx

                if best_iou >= IOU_THRESH and best_gt_idx != -1:
                    gts[best_gt_idx]["matched"] = True
                    tp_total += 1
                    if p["cls"] == 0:
                        tp_milco += 1
                    else:
                        tp_nombo += 1
                else:
                    fp_total += 1
                    if p["cls"] == 0:
                        fp_milco += 1
                    else:
                        fp_nombo += 1

            if is_empty_seabed and len(filtered_preds) > 0:
                fp_empty_seabed += len(filtered_preds)

            for g in gts:
                if not g["matched"]:
                    fn_total += 1
                    if g["cls"] == 0:
                        fn_milco += 1
                    else:
                        fn_nombo += 1

        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        results_table.append({
            "Confidence Threshold": f"{conf:.2f}",
            "Precision": f"{precision:.4f} ({precision*100:.2f}%)",
            "Recall": f"{recall:.4f} ({recall*100:.2f}%)",
            "F1": f"{f1:.4f}",
            "True Positives (TP)": tp_total,
            "False Positives (FP)": fp_total,
            "False Negatives (FN)": fn_total,
            "TP (MILCO / NOMBO)": f"{tp_milco} / {tp_nombo}",
            "FP (MILCO / NOMBO)": f"{fp_milco} / {fp_nombo}",
            "FP on Empty Seabed": fp_empty_seabed,
            "FN (MILCO / NOMBO)": f"{fn_milco} / {fn_nombo}"
        })

    df = pd.DataFrame(results_table)
    print("\n" + "=" * 90)
    print("YOLO11s CONFIDENCE THRESHOLD COMPARISON ON 2018 UNSEEN SURVEY (564 IMAGES)")
    print("=" * 90)
    print(df[["Confidence Threshold", "Precision", "Recall", "F1", "False Positives (FP)", "False Negatives (FN)"]].to_string(index=False))
    print("=" * 90)

    out_csv = Path("milco_dataset/error_analysis_yolo11s/confidence_threshold_sweep.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[✓] Sweep results exported to: {out_csv.resolve()}")

if __name__ == "__main__":
    main()
