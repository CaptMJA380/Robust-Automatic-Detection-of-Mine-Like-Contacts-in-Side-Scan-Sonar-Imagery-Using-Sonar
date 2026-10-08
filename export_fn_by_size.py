import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def categorize_size(area_px):
    if area_px < 256:        # < 16x16 px
        return "Tiny"
    elif area_px < 1024:     # 16x16 to 32x32 px
        return "Small"
    elif area_px < 9216:     # 32x32 to 96x96 px
        return "Medium"
    else:                    # >= 96x96 px
        return "Large"

def main():
    model_path = Path("runs/detect/yolo11s_primary_100epochs/weights/best.pt")
    model = YOLO(str(model_path))
    test_img_dir = Path("milco_dataset/yolo_primary/images/test")
    test_lbl_dir = Path("milco_dataset/yolo_primary/labels/test")

    img_paths = sorted(list(test_img_dir.glob("*.jpg")))
    records = []

    for img_p in img_paths:
        stem = img_p.stem
        lbl_p = test_lbl_dir / f"{stem}.txt"
        if not lbl_p.exists():
            continue
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]

        gts = []
        for line in lbl_p.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = (xc - w / 2.0) * w_img
                y1 = (yc - h / 2.0) * h_img
                x2 = (xc + w / 2.0) * w_img
                y2 = (yc + h / 2.0) * h_img
                area_px = (x2 - x1) * (y2 - y1)
                gts.append({
                    "image": stem,
                    "cls": cls_id,
                    "cls_name": "MILCO" if cls_id == 0 else "NOMBO",
                    "box": [x1, y1, x2, y2],
                    "area_px": area_px,
                    "size_category": categorize_size(area_px),
                    "matched": False
                })

        if not gts:
            continue

        res = model.predict(source=str(img_p), conf=0.25, imgsz=640, device=0, verbose=False)[0]
        preds = [{"cls": int(b.cls[0].item()), "box": b.xyxy[0].tolist()} for b in res.boxes]

        for p in preds:
            best_iou = 0.0
            best_idx = -1
            for idx, g in enumerate(gts):
                if not g["matched"] and g["cls"] == p["cls"]:
                    iou = compute_iou(p["box"], g["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx
            if best_iou >= 0.50 and best_idx != -1:
                gts[best_idx]["matched"] = True

        for g in gts:
            records.append(g)

    df = pd.DataFrame(records)
    fn_df = df[~df["matched"]]

    out_csv = Path("milco_dataset/error_analysis_yolo11s/false_negatives_by_size.csv")
    fn_df.to_csv(out_csv, index=False)
    print(f"Exported {len(fn_df)} False Negatives by size to: {out_csv.resolve()}")

if __name__ == "__main__":
    main()
