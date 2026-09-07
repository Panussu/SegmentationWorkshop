"""รันการทดลองทั้งหมด: ปรับ threshold -> ทดสอบ -> บันทึกผลและกราฟ."""
from pathlib import Path
import csv
import json

import cv2
import matplotlib
matplotlib.use("Agg")  # บันทึกกราฟได้โดยไม่ต้องเปิดหน้าต่าง
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score

from segmentation import foreground_score, clean_mask, confusion_counts, metrics

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"


def load_sample(name):
    """ย่อภาพให้ด้านยาวไม่เกิน 256 และอ่าน trimap โดยรักษาค่า label."""
    with Image.open(DATA / "images" / (name + ".jpg")) as image:
        rgb = np.array(image.convert("RGB"))
    with Image.open(DATA / "masks" / (name + ".png")) as image:
        trimap = np.array(image)
    if trimap.shape != rgb.shape[:2] or not set(np.unique(trimap)) <= {1, 2, 3}:
        raise ValueError(f"Invalid trimap: {name}")
    height, width = rgb.shape[:2]
    scale = min(1.0, 256 / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    trimap = cv2.resize(trimap, size, interpolation=cv2.INTER_NEAREST)
    # 1=สัตว์, 2=พื้นหลัง, 3=บริเวณขอบที่ไม่แน่นอน (ไม่นำมาคิดคะแนน)
    return rgb, trimap == 1, trimap != 3


def choose_threshold(names):
    """เลือก threshold จาก validation เท่านั้น โดยใช้ mean IoU ก่อน Morphology."""
    samples = []
    for name in names:
        rgb, truth, valid = load_sample(name)
        samples.append((foreground_score(rgb), truth, valid))
    candidates = []
    for threshold in np.arange(0.10, 0.91, 0.05):
        scores = [metrics(confusion_counts(truth[valid], (score >= threshold)[valid]))["iou"]
                  for score, truth, valid in samples]
        candidates.append({"threshold": round(float(threshold), 2), "mean_iou": float(np.mean(scores))})
    return max(candidates, key=lambda item: item["mean_iou"])["threshold"], candidates


def save_example(name, rgb, truth, valid, raw, cleaned):
    """แสดงภาพจริง เฉลย และ mask ก่อน/หลัง โดยระบายขอบที่ไม่ประเมินเป็นสีเทา."""
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    gt = np.where(valid, truth.astype(float), 0.5)
    cutout = rgb.copy()
    cutout[~cleaned] = 0
    for ax, title, picture in zip(axes, ["Input", "GT (gray = ignored)", "Before", "After", "Cutout"],
                                  [rgb, gt, raw, cleaned, cutout]):
        ax.imshow(picture, cmap="gray", vmin=0, vmax=1) if picture.ndim == 2 else ax.imshow(picture)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(OUT / "examples" / (name + ".png"), dpi=120)
    plt.close(fig)


def main():
    if not (DATA / "split.json").exists():
        raise SystemExit("Run python download_data.py first.")
    split = json.loads((DATA / "split.json").read_text())
    if len(split["test"]) < 50 or set(split["validation"]) & set(split["test"]):
        raise ValueError("Need at least 50 test images and disjoint splits.")
    (OUT / "examples").mkdir(parents=True, exist_ok=True)
    threshold, tuning = choose_threshold(split["validation"])
    print(f"Validation selected threshold: {threshold}", flush=True)
    totals = {stage: dict(TN=0, FP=0, FN=0, TP=0) for stage in ("before", "after")}
    rows, all_truth, all_scores = [], [], []
    for index, name in enumerate(split["test"]):
        rgb, truth, valid = load_sample(name)
        score = foreground_score(rgb)
        raw = score >= threshold
        cleaned = clean_mask(raw)
        for stage, prediction in [("before", raw), ("after", cleaned)]:
            counts = confusion_counts(truth[valid], prediction[valid])
            for key, count in counts.items():
                totals[stage][key] += count
            rows.append({"image": name, "stage": stage, **counts, **metrics(counts)})
        all_truth.append(truth[valid])
        all_scores.append(score[valid])
        # เก็บตัวอย่างทุกภาพ ไม่เลือกเฉพาะภาพที่ผลดี
        save_example(name, rgb, truth, valid, raw, cleaned)
        print(f"Evaluated {index + 1}/{len(split['test'])}: {name}", flush=True)
    y_true, y_score = np.concatenate(all_truth), np.concatenate(all_scores)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = float(roc_auc_score(y_true, y_score))
    summary = {
        "validation_images": len(split["validation"]), "test_images": len(split["test"]),
        "threshold": threshold, "max_image_side": 256, "ignored_trimap_label": 3,
        "evaluated_pixels": int(y_true.size), "score_roc_auc": auc,
        **{stage: {**counts, **metrics(counts)} for stage, counts in totals.items()},
    }
    for filename, content in [("summary.json", summary), ("threshold_search.json", tuning), ("split.json", split)]:
        (OUT / filename).write_text(json.dumps(content, indent=2), encoding="utf-8")
    with (OUT / "per_image.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (stage, counts) in zip(axes, totals.items()):
        matrix = np.array([[counts["TN"], counts["FP"]], [counts["FN"], counts["TP"]]])
        ax.imshow(matrix, cmap="Blues")
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, f"{value:,}", ha="center", va="center",
                    color="white" if value > matrix.max() / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Background", "Pet"],
               yticklabels=["Background", "Pet"], xlabel="Predicted", ylabel="Actual", title=stage.title())
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"Color-distance score (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    # Morphology ให้ binary mask จึงแสดงเป็นจุดทำงาน ไม่สร้าง ROC จาก mask 0/1
    for stage, counts in totals.items():
        fp_rate = counts["FP"] / (counts["FP"] + counts["TN"])
        ax.scatter(fp_rate, metrics(counts)["recall"], label=stage.title(), s=45)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Test pixels: ROC", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "roc_curve.png", dpi=150)
    plt.close(fig)
    lines = ["# Experiment results", "", "Generated by `python run.py` from real Oxford-IIIT Pet images.", "",
             f"Validation: {len(split['validation'])} images. Test: {len(split['test'])} images. Threshold: {threshold}.",
             "", "Metrics pool all valid test pixels after resizing; trimap label 3 is excluded.", "",
             "| Stage | Accuracy | Precision | Recall | IoU | Dice |", "|---|---:|---:|---:|---:|---:|"]
    for stage in totals:
        values = metrics(totals[stage])
        lines.append("| " + stage + " | " + " | ".join(f"{v:.4f}" for v in values.values()) + " |")
    lines += ["", f"Continuous score ROC AUC: {auc:.4f}.", "",
              "Morphology is shown as an operating point, not a separate ROC curve.", "",
              "This simple baseline assumes the border represents background. Similar pet/background colors and pets touching the border can cause errors. Morphology can remove small details and does not guarantee an improvement."]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
