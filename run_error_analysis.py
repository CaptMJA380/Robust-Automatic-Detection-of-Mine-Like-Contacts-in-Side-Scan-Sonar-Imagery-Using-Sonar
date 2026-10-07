import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

def compute_iou(box1, box2):
    # box format: [x1, y1, x2, y2]
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

def compute_local_contrast(img, box):
    # box: [x1, y1, x2, y2] in pixel coords
    h, w = img.shape[:2]
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    x1, y1 = max(0, x1), max(0, y1)
    xB, yB = min(w, x2), min(h, y2)
    
    if xB <= x1 or yB <= y1:
        return 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    target_patch = gray[y1:yB, x1:xB]
    
    # Context window around the target (2x width/height)
    pad_w = (xB - x1) // 2
    pad_h = (yB - y1) // 2
    cx1 = max(0, x1 - pad_w)
    cy1 = max(0, y1 - pad_h)
    cx2 = min(w, xB + pad_w)
    cy2 = min(h, yB + pad_h)
    context_patch = gray[cy1:cy2, cx1:cx2]

    # Michelson contrast or RMS contrast
    fg_mean = float(np.mean(target_patch))
    bg_mean = float(np.mean(context_patch))
    denom = fg_mean + bg_mean
    contrast = abs(fg_mean - bg_mean) / (denom + 1e-5)
    return contrast

def main():
    print("=" * 80)
    print("RUNNING ERROR ANALYSIS ON 2018 UNSEEN TEST SURVEY")
    print("=" * 80)

    model_path = Path("runs/detect/yolov8s_primary_100epochs/weights/best.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing weights: {model_path}")

    model = YOLO(str(model_path))
    test_img_dir = Path("milco_dataset/yolo_primary/images/test")
    test_lbl_dir = Path("milco_dataset/yolo_primary/labels/test")
    
    out_dir = Path("milco_dataset/error_analysis")
    gallery_dir = out_dir / "representative_examples"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(list(test_img_dir.glob("*.jpg")))
    print(f"Total test images: {len(img_paths)}")

    # Class definitions
    CLASS_NAMES = {0: "MILCO", 1: "NOMBO"}

    # Detection threshold
    CONF_THRESH = 0.25
    IOU_THRESH = 0.50

    # Metrics counters
    tp_milco = 0
    tp_nombo = 0
    fp_milco = 0
    fp_nombo = 0
    fp_empty_seabed = 0
    fn_milco = 0
    fn_nombo = 0

    all_records = []
    
    # Categorized candidate pools for visualization
    candidates = {
        "correct_milco": [],
        "missed_milco": [],
        "correct_nombo": [],
        "missed_nombo": [],
        "fp_empty_seabed": [],
        "tiny_target_missed": [],
        "low_contrast_target_missed": [],
        "all_fp": [],
        "all_fn": []
    }

    for img_p in img_paths:
        stem = img_p.stem
        lbl_p = test_lbl_dir / f"{stem}.txt"
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]
        img_area = h_img * w_img

        # 1. Parse ground truth
        gt_boxes = []  # list of dict: {cls, box: [x1, y1, x2, y2], matched: False, area, contrast}
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
                    box = [x1, y1, x2, y2]
                    area_px = (x2 - x1) * (y2 - y1)
                    contrast = compute_local_contrast(img, box)
                    gt_boxes.append({
                        "cls": cls_id,
                        "box": box,
                        "matched": False,
                        "area_px": area_px,
                        "contrast": contrast
                    })

        is_empty_seabed = (len(gt_boxes) == 0)

        # 2. Run model prediction
        pred_res = model.predict(source=str(img_p), conf=CONF_THRESH, iou=0.45, device=0, verbose=False)[0]
        pred_boxes = [] # list of dict: {cls, conf, box: [x1, y1, x2, y2], matched: False}
        if pred_res.boxes is not None and len(pred_res.boxes) > 0:
            for b in pred_res.boxes:
                cls_id = int(b.cls.item())
                conf = float(b.conf.item())
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                pred_boxes.append({
                    "cls": cls_id,
                    "conf": conf,
                    "box": xyxy,
                    "matched": False
                })

        # 3. Match Predictions to Ground Truth
        # Greedily match highest confidence predictions first
        pred_boxes.sort(key=lambda x: x["conf"], reverse=True)

        for p in pred_boxes:
            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, g in enumerate(gt_boxes):
                if not g["matched"] and p["cls"] == g["cls"]:
                    iou = compute_iou(p["box"], g["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

            if best_iou >= IOU_THRESH and best_gt_idx >= 0:
                # True Positive
                p["matched"] = True
                gt_boxes[best_gt_idx]["matched"] = True
                if p["cls"] == 0:
                    tp_milco += 1
                    candidates["correct_milco"].append({"img_p": img_p, "pred": p, "gt": gt_boxes[best_gt_idx], "iou": best_iou})
                else:
                    tp_nombo += 1
                    candidates["correct_nombo"].append({"img_p": img_p, "pred": p, "gt": gt_boxes[best_gt_idx], "iou": best_iou})
            else:
                # False Positive
                if p["cls"] == 0:
                    fp_milco += 1
                else:
                    fp_nombo += 1
                
                fp_info = {"img_p": img_p, "pred": p, "is_empty_seabed": is_empty_seabed, "all_gt": gt_boxes}
                candidates["all_fp"].append(fp_info)
                if is_empty_seabed:
                    fp_empty_seabed += 1
                    candidates["fp_empty_seabed"].append(fp_info)

        # 4. Check Unmatched Ground Truth (False Negatives / Misses)
        for g in gt_boxes:
            if not g["matched"]:
                fn_info = {"img_p": img_p, "gt": g, "all_pred": pred_boxes}
                candidates["all_fn"].append(fn_info)
                if g["cls"] == 0:
                    fn_milco += 1
                    candidates["missed_milco"].append(fn_info)
                else:
                    fn_nombo += 1
                    candidates["missed_nombo"].append(fn_info)

                # Check special error types
                if g["area_px"] < 32 * 32:  # Tiny target
                    candidates["tiny_target_missed"].append(fn_info)
                if g["contrast"] < 0.12:    # Low contrast target
                    candidates["low_contrast_target_missed"].append(fn_info)

    # Print Collected Counts
    total_tp = tp_milco + tp_nombo
    total_fp = fp_milco + fp_nombo
    total_fn = fn_milco + fn_nombo

    print("\n" + "=" * 80)
    print(f"ERROR ANALYSIS RESULTS ON 2018 TEST SET (conf_thresh >= {CONF_THRESH}, IoU >= {IOU_THRESH})")
    print("=" * 80)
    print(f"Total Test Images           : {len(img_paths)} (Empty Seabed: 452, Target Scenes: 112)")
    print(f"Total Ground Truth Contacts : {total_tp + total_fn} (MILCO: {tp_milco + fn_milco}, NOMBO: {tp_nombo + fn_nombo})")
    print("-" * 80)
    print(f"True Positives (TP) Total   : {total_tp}  (MILCO: {tp_milco}, NOMBO: {tp_nombo})")
    print(f"False Positives (FP) Total  : {total_fp}  (MILCO: {fp_milco}, NOMBO: {fp_nombo})")
    print(f"  - FP on Empty Seabed Scenes : {fp_empty_seabed}  ({(fp_empty_seabed / max(1, total_fp))*100:.1f}% of all false alarms)")
    print(f"False Negatives (FN) Total  : {total_fn}  (MILCO Missed: {fn_milco}, NOMBO Missed: {fn_nombo})")
    print("-" * 80)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"Operational Precision (P)   : {precision:.4f} ({precision*100:.2f}%)")
    print(f"Operational Recall (R)      : {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score                    : {f1:.4f}")
    print("=" * 80)

    # Helper function to draw clear professional annotations
    def draw_annotated_card(img_raw, title, gts, preds, highlight_box=None, highlight_type="FN"):
        vis = img_raw.copy()
        h, w = vis.shape[:2]

        # Draw all ground truths in GREEN dashed/solid
        for g in gts:
            b = [int(v) for v in g["box"]]
            cls_name = CLASS_NAMES.get(g["cls"], "Unknown")
            cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            cv2.putText(vis, f"GT: {cls_name}", (b[0], max(15, b[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw all predictions in RED / CYAN
        for p in preds:
            b = [int(v) for v in p["box"]]
            cls_name = CLASS_NAMES.get(p["cls"], "Unknown")
            col = (0, 0, 255) if p["cls"] == 0 else (255, 128, 0) # Red for MILCO, Orange for NOMBO
            cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), col, 2)
            cv2.putText(vis, f"PRED: {cls_name} ({p['conf']:.2f})", (b[0], min(h - 8, b[3] + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        # Draw a prominent banner on top with category title
        banner_h = 45
        banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
        # Background banner color depending on type
        if "Correct" in title:
            bg_col = (30, 100, 30) # Dark green
        elif "Empty Seabed" in title or "False Positive" in title:
            bg_col = (30, 30, 150) # Dark red
        else:
            bg_col = (140, 60, 20) # Indigo / purple
        banner[:] = bg_col

        cv2.putText(banner, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        combined = np.vstack([banner, vis])
        return combined

    # 1. Save the 7 Representative Images Requested
    print("\n[>>] Generating 7 Representative Analysis Images...")

    rep_categories = [
        ("1. Correct MILCO", "correct_milco", "1_correct_milco.png"),
        ("2. Missed MILCO", "missed_milco", "2_missed_milco.png"),
        ("3. Correct NOMBO", "correct_nombo", "3_correct_nombo.png"),
        ("4. Missed NOMBO", "missed_nombo", "4_missed_nombo.png"),
        ("5. False Positive on Empty Seabed", "fp_empty_seabed", "5_false_positive_empty_seabed.png"),
        ("6. Tiny Target Missed", "tiny_target_missed", "6_tiny_target_missed.png"),
        ("7. Low-Contrast Target Missed", "low_contrast_target_missed", "7_low_contrast_target_missed.png")
    ]

    saved_rep_paths = []
    for label, cat_key, fname in rep_categories:
        pool = candidates[cat_key]
        if not pool:
            print(f"[!] Warning: No candidate found for {label}")
            continue
        # Pick the most illustrative example (e.g. median or highest confidence/contrast)
        item = pool[0]
        img_raw = cv2.imread(str(item["img_p"]))
        
        if "correct" in cat_key:
            gts = [item["gt"]]
            preds = [item["pred"]]
        elif "fp" in cat_key:
            gts = item["all_gt"]
            preds = [item["pred"]]
        else: # missed
            gts = [item["gt"]]
            preds = item["all_pred"]

        card = draw_annotated_card(img_raw, f"{label} [{item['img_p'].name}]", gts, preds)
        out_path = out_dir / fname
        cv2.imwrite(str(out_path), card)
        saved_rep_paths.append(out_path)
        print(f" [✓] Saved {label} -> {out_path.name}")

    # 2. Save 25-30 False Negative and False Positive Examples to Gallery
    print("\n[>>] Curating 26 False-Negative and False-Positive Gallery Examples...")
    gallery_count = 0

    # 13 False Positives
    for idx, item in enumerate(candidates["all_fp"][:13]):
        gallery_count += 1
        img_raw = cv2.imread(str(item["img_p"]))
        is_empty = item["is_empty_seabed"]
        p_cls = CLASS_NAMES.get(item["pred"]["cls"])
        tag = f"FP_{gallery_count:02d}_Pred_{p_cls}_{item['pred']['conf']:.2f}_{'EmptySeabed' if is_empty else 'Clutter'}"
        title = f"False Positive #{gallery_count:02d}: Predicted {p_cls} ({item['pred']['conf']:.2f}) on {'Empty Seabed' if is_empty else 'Non-empty scene'}"
        card = draw_annotated_card(img_raw, title, item["all_gt"], [item["pred"]])
        cv2.imwrite(str(gallery_dir / f"{tag}_{item['img_p'].stem}.jpg"), card)

    # 13 False Negatives
    for idx, item in enumerate(candidates["all_fn"][:13]):
        gallery_count += 1
        img_raw = cv2.imread(str(item["img_p"]))
        gt_cls = CLASS_NAMES.get(item["gt"]["cls"])
        area_px = item["gt"]["area_px"]
        contrast = item["gt"]["contrast"]
        tag = f"FN_{gallery_count:02d}_Missed_{gt_cls}_Area{int(area_px)}px_Contrast{contrast:.2f}"
        title = f"False Negative #{gallery_count:02d}: Missed {gt_cls} (Area: {int(area_px)}px, Contrast: {contrast:.2f})"
        card = draw_annotated_card(img_raw, title, [item["gt"]], item["all_pred"])
        cv2.imwrite(str(gallery_dir / f"{tag}_{item['img_p'].stem}.jpg"), card)

    print(f" [✓] Successfully saved {gallery_count} gallery images into: {gallery_dir.resolve()}")

    # 3. Save Summary CSV Report
    summary_csv = out_dir / "error_analysis_summary.csv"
    summary_df = pd.DataFrame([
        {"Category": "True Positives (MILCO)", "Count": tp_milco},
        {"Category": "True Positives (NOMBO)", "Count": tp_nombo},
        {"Category": "Total True Positives", "Count": total_tp},
        {"Category": "False Positives (MILCO)", "Count": fp_milco},
        {"Category": "False Positives (NOMBO)", "Count": fp_nombo},
        {"Category": "False Positives on Empty Seabed", "Count": fp_empty_seabed},
        {"Category": "Total False Positives", "Count": total_fp},
        {"Category": "False Negatives (Missed MILCO)", "Count": fn_milco},
        {"Category": "False Negatives (Missed NOMBO)", "Count": fn_nombo},
        {"Category": "Total False Negatives", "Count": total_fn},
        {"Category": "Operational Precision", "Count": round(precision, 4)},
        {"Category": "Operational Recall", "Count": round(recall, 4)},
        {"Category": "Operational F1-Score", "Count": round(f1, 4)},
    ])
    summary_df.to_csv(summary_csv, index=False)
    print(f" [✓] Error analysis summary CSV saved to: {summary_csv.resolve()}")

if __name__ == "__main__":
    main()
