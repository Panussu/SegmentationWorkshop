# =============================================================================
# ส่วนที่ 1: คำอธิบายโมดูลและไลบรารีที่ใช้
# =============================================================================
# ระบุว่าโมดูลนี้ควบคุมขั้นตอนการทดลองตั้งแต่ปรับ Threshold จนถึงบันทึกผลลัพธ์
"""รันการทดลองทั้งหมด: ปรับ threshold -> ทดสอบ -> บันทึกผลและกราฟ."""
# นำเข้า Path สำหรับสร้างและจัดการพาธไฟล์
from pathlib import Path
# นำเข้า argparse สำหรับจัดการพารามิเตอร์ผ่าน Command line
import argparse
# นำเข้า csv สำหรับเขียนผลประเมินแยกตามรูปภาพ
import csv
# นำเข้า json สำหรับอ่าน Split และเขียนผลสรุป
import json
# ใช้ตัวเขียน JSON ที่คงคำอธิบายภาษาไทยไว้เมื่อรันใหม่
from json_comments import write_commented_json

# นำเข้า OpenCV สำหรับย่อภาพและ Mask
import cv2
# นำเข้า Matplotlib เพื่อสร้างกราฟและภาพผลลัพธ์
import matplotlib
# เลือก Backend แบบไม่เปิดหน้าต่าง เพื่อให้บันทึกกราฟบนเครื่องที่ไม่มีหน้าจอได้
matplotlib.use("Agg")
# นำเข้า pyplot สำหรับสร้าง Figure และ Axes
import matplotlib.pyplot as plt
# นำเข้า NumPy สำหรับประมวลผลอาร์เรย์ภาพและคำนวณทางคณิตศาสตร์
import numpy as np
# นำเข้า Image จาก Pillow สำหรับเปิดไฟล์ภาพและ Trimap
from PIL import Image
# นำเข้าฟังก์ชันสร้าง ROC curve และคำนวณ ROC AUC จาก scikit-learn
from sklearn.metrics import roc_curve, roc_auc_score, auc as sk_auc

# นำเข้าฟังก์ชัน Segmentation และฟังก์ชันประเมินผลจากโมดูลภายในโครงการ
from segmentation import foreground_score, clean_mask, confusion_counts, metrics

# =============================================================================
# ส่วนที่ 2: การกำหนดพาธของชุดข้อมูลและผลลัพธ์
# =============================================================================
# หาพาธโฟลเดอร์รากจากตำแหน่งของไฟล์ Python ปัจจุบัน
ROOT = Path(__file__).resolve().parent
# กำหนดพาธโฟลเดอร์ที่เก็บชุดข้อมูล
DATA = ROOT / "data"
# กำหนดพาธโฟลเดอร์ที่ใช้บันทึกผลลัพธ์ทั้งหมด
OUT = ROOT / "outputs"


# =============================================================================
# ส่วนที่ 2.1: การจัดการค่าคอนฟิกและพารามิเตอร์การทดลอง (CLI & JSON Config)
# =============================================================================
# ประกาศฟังก์ชันสำหรับประมวลผลพารามิเตอร์จาก Command line หรือไฟล์ config.json
def parse_args():
    """ประมวลผลพารามิเตอร์จาก Command-line Arguments หรือไฟล์ config.json."""
    # ตรวจสอบเบื้องต้นว่ามีการระบุ --config หรือไม่
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    config_defaults = {}
    if pre_args.config:
        if not pre_args.config.exists():
            raise SystemExit(f"Config file not found: {pre_args.config}")
        cfg = json.loads(pre_args.config.read_text(encoding="utf-8"))
        config_defaults = {k.replace("-", "_"): v for k, v in cfg.items() if k != "_comment"}
        if "data_dir" in config_defaults:
            config_defaults["data_dir"] = Path(config_defaults["data_dir"])
        if "output_dir" in config_defaults:
            config_defaults["output_dir"] = Path(config_defaults["output_dir"])

    parser = argparse.ArgumentParser(
        description="รันการทดลอง Image Segmentation: ปรับ threshold -> ทดสอบ -> บันทึกผลและกราฟ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=None, help="พาธของไฟล์ config.json สำหรับโหลดการตั้งค่าทั้งหมด")
    parser.add_argument("--data-dir", type=Path, default=DATA, help="พาธโฟลเดอร์ที่เก็บชุดข้อมูล")
    parser.add_argument("--output-dir", type=Path, default=OUT, help="พาธโฟลเดอร์ที่ใช้บันทึกผลลัพธ์")
    parser.add_argument("--max-image-side", type=int, default=256, help="ความยาวด้านที่ยาวที่สุดหลังย่อภาพ (พิกเซล)")
    parser.add_argument("--threshold-min", type=float, default=0.03, help="ค่า Threshold ต่ำสุดในการค้นหา")
    parser.add_argument("--threshold-max", type=float, default=0.90, help="ค่า Threshold สูงสุดในการค้นหา")
    parser.add_argument("--threshold-step", type=float, default=0.02, help="สเต็ปในการค้นหา Threshold")
    parser.add_argument("--fixed-threshold", type=float, default=None, help="ระบุ Threshold คงที่โดยตรง ข้ามขั้นตอนจูนบน Validation")
    parser.add_argument("--kernel-size", type=int, default=3, help="ขนาด Kernel สำหรับ Morphology ใน clean_mask (เลขคี่)")
    parser.add_argument("--distance-scale", type=float, default=20.0, help="ตัวหารปรับสเกลคะแนนระยะสีใน foreground_score (CIELAB a*)")

    if config_defaults:
        parser.set_defaults(**config_defaults)

    return parser.parse_args()


# =============================================================================
# ส่วนที่ 3: การโหลดและเตรียมภาพกับ Ground truth
# =============================================================================
# ประกาศฟังก์ชันสำหรับโหลดรูปภาพและ Ground truth หนึ่งตัวอย่าง
def load_sample(name, data_dir=DATA, max_image_side=256):
    # อธิบายว่าฟังก์ชันย่อภาพและรักษาค่า Label ของ Trimap
    """ย่อภาพให้ด้านยาวไม่เกิน max_image_side และอ่าน trimap โดยรักษาค่า label."""
    data_dir = Path(data_dir)
    # เปิดไฟล์ภาพสีตามชื่อที่รับเข้ามาและปิดไฟล์อัตโนมัติเมื่ออ่านเสร็จ
    with Image.open(data_dir / "images" / (name + ".jpg")) as image:
        # แปลงภาพเป็น RGB แล้วเปลี่ยนเป็นอาร์เรย์ NumPy
        rgb = np.array(image.convert("RGB"))
    # เปิดไฟล์ Trimap ตามชื่อภาพและปิดไฟล์อัตโนมัติเมื่ออ่านเสร็จ
    with Image.open(data_dir / "masks" / (name + ".png")) as image:
        # แปลง Trimap เป็นอาร์เรย์ NumPy โดยคงค่า Label เดิม
        trimap = np.array(image)
    # ตรวจสอบว่าขนาด Trimap ตรงกับภาพและมีเฉพาะ Label 1, 2 และ 3
    if trimap.shape != rgb.shape[:2] or not set(np.unique(trimap)) <= {1, 2, 3}:
        # แจ้งข้อผิดพลาดพร้อมชื่อไฟล์เมื่อ Trimap ไม่ถูกต้อง
        raise ValueError(f"Invalid trimap: {name}")
    # อ่านความสูงและความกว้างของภาพจากสองมิติแรก
    height, width = rgb.shape[:2]
    # คำนวณอัตราย่อโดยให้ด้านยาวไม่เกิน max_image_side พิกเซลและไม่ขยายภาพเล็ก
    scale = min(1.0, max_image_side / max(height, width))
    # คำนวณขนาดใหม่โดยรับประกันว่าแต่ละด้านมีอย่างน้อย 1 พิกเซล
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    # ย่อภาพสีด้วย Area interpolation ซึ่งเหมาะกับการลดขนาดภาพ
    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    # ย่อ Trimap ด้วย Nearest-neighbor เพื่อไม่ให้เกิดค่า Label ใหม่
    trimap = cv2.resize(trimap, size, interpolation=cv2.INTER_NEAREST)
    # อธิบายความหมายของแต่ละ Label และการละเว้นบริเวณขอบที่ไม่แน่นอน
    # 1=แอปเปิลแดง, 2=พื้นหลังและเงา, 3=บริเวณขอบที่ไม่แน่นอน (ไม่นำมาคิดคะแนน)
    # คืนภาพ RGB, Mask วัตถุ และ Mask ระบุพิกเซลที่ใช้ประเมินผล
    return rgb, trimap == 1, trimap != 3


# =============================================================================
# ส่วนที่ 4: การค้นหา Threshold ที่เหมาะสมจากชุด Validation
# =============================================================================
# ประกาศฟังก์ชันสำหรับเลือก Threshold จากชุด Validation (ทั้งก่อนและหลังทำ Morphology)
def choose_threshold(names, data_dir=DATA, max_image_side=256, threshold_min=0.10, threshold_max=0.90, threshold_step=0.02, distance_scale=20.0, kernel_size=3):
    # อธิบายว่าฟังก์ชันเลือก Threshold ด้วยค่าเฉลี่ย IoU บนชุด Validation
    """เลือก threshold ที่ดีที่สุดสำหรับ Before และ After Morphology จากชุด Validation."""
    # สร้าง List ว่างสำหรับเก็บคะแนน Ground truth และ Valid mask ของทุกภาพ
    samples = []
    # วนอ่านภาพ Validation แต่ละชื่อ
    for name in names:
        # โหลดภาพ Ground truth และบริเวณที่ใช้ประเมินของตัวอย่างปัจจุบัน
        rgb, truth, valid = load_sample(name, data_dir=data_dir, max_image_side=max_image_side)
        # คำนวณคะแนนวัตถุแล้วเก็บพร้อม Ground truth และ Valid mask
        samples.append((foreground_score(rgb, distance_scale=distance_scale), truth, valid))
    # สร้าง List ว่างสำหรับเก็บผลของ Threshold แต่ละค่า
    candidates = []
    # วน Threshold ตามช่วงและสเต็ปที่กำหนด
    steps = np.arange(threshold_min, threshold_max + threshold_step / 2.0, threshold_step)
    for threshold in steps:
        # คำนวณ IoU ก่อน Morphology
        scores_before = [metrics(confusion_counts(truth[valid], (score >= threshold)[valid]))["iou"] for score, truth, valid in samples]
        # คำนวณ IoU หลัง Morphology
        scores_after = [metrics(confusion_counts(truth[valid], clean_mask((score >= threshold), kernel_size=kernel_size)[valid]))["iou"] for score, truth, valid in samples]
        candidates.append({
            "threshold": round(float(threshold), 2),
            "mean_iou_before": float(np.mean(scores_before)),
            "mean_iou_after": float(np.mean(scores_after))
        })
    best_before = max(candidates, key=lambda item: item["mean_iou_before"])["threshold"]
    best_after = max(candidates, key=lambda item: item["mean_iou_after"])["threshold"]
    return best_before, best_after, candidates


# =============================================================================
# ส่วนที่ 5: การสร้างภาพตัวอย่างเปรียบเทียบก่อนและหลัง Morphology
# =============================================================================
# ประกาศฟังก์ชันสำหรับบันทึกภาพเปรียบเทียบผลของหนึ่งตัวอย่าง
def save_example(name, rgb, truth, valid, raw, cleaned, output_dir=OUT):
    # อธิบายองค์ประกอบทั้งห้าที่จะแสดงในภาพเปรียบเทียบ
    """แสดงภาพจริง เฉลย และ mask ก่อน/หลัง โดยระบายขอบที่ไม่ประเมินเป็นสีเทา."""
    output_dir = Path(output_dir)
    # สร้าง Figure ที่มีกราฟย่อยหนึ่งแถวจำนวนห้าช่อง
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    # สร้างภาพ Ground truth ที่กำหนดบริเวณไม่ใช้ประเมินเป็นสีเทาค่า 0.5
    gt = np.where(valid, truth.astype(float), 0.5)
    # สำเนาภาพ RGB เพื่อสร้างภาพที่ตัดพื้นหลังออก
    cutout = rgb.copy()
    # เปลี่ยนพิกเซลที่ Mask หลังทำความสะอาดระบุเป็นพื้นหลังให้เป็นสีดำ
    cutout[~cleaned] = 0
    # วนจับคู่ Axes ชื่อหัวข้อ และภาพทั้งห้ารูปตามลำดับ
    for ax, title, picture in zip(axes, ["Input", "GT (gray = ignored)", "Before", "After", "Cutout"], [rgb, gt, raw, cleaned, cutout]):
        # แสดงภาพแบบ Grayscale เมื่อมีสองมิติ หรือแสดงภาพสีเมื่อมีสามมิติ
        ax.imshow(picture, cmap="gray", vmin=0, vmax=1) if picture.ndim == 2 else ax.imshow(picture)
        # กำหนดชื่อหัวข้อของกราฟย่อยปัจจุบัน
        ax.set_title(title)
        # ซ่อนแกนและค่าพิกัดเพื่อให้เห็นภาพชัดเจน
        ax.axis("off")
    # กำหนดชื่อภาพตัวอย่างเป็นหัวข้อหลักของ Figure
    fig.suptitle(name)
    # ปรับระยะห่างขององค์ประกอบใน Figure อัตโนมัติ
    fig.tight_layout()
    # บันทึก Figure ลงโฟลเดอร์ examples ด้วยความละเอียด 120 DPI
    fig.savefig(output_dir / "examples" / (name + ".png"), dpi=120)
    # ปิด Figure เพื่อคืนหน่วยความจำหลังบันทึกไฟล์
    plt.close(fig)


# =============================================================================
# ส่วนที่ 6: กระบวนการทดลองหลักและการตรวจสอบชุดข้อมูล
# =============================================================================
# ประกาศฟังก์ชันหลักที่ควบคุมการทดลองทั้งหมด
def main(args=None):
    # หากไม่ได้ส่ง args เข้ามา ให้ประมวลผลจาก Command line
    if args is None:
        args = parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)

    # ตรวจสอบว่ามีไฟล์แบ่งชุดข้อมูลที่สร้างโดย download_data.py แล้วหรือไม่
    if not (data_dir / "split.json").exists():
        # หยุดโปรแกรมพร้อมแจ้งให้ดาวน์โหลดข้อมูลก่อนเมื่อไม่พบไฟล์ Split
        raise SystemExit(f"Run python download_data.py first. Missing: {data_dir / 'split.json'}")
    # อ่านข้อความ JSON ของ Split แล้วแปลงเป็น Dictionary
    split = json.loads((data_dir / "split.json").read_text(encoding="utf-8"))
    # ตรวจสอบว่ามีภาพ Test อย่างน้อย 50 ภาพและไม่มีภาพซ้ำกับ Validation
    if len(split["test"]) < 50 or set(split["validation"]) & set(split["test"]):
        # แจ้งข้อผิดพลาดเมื่อขนาดหรือการแยกชุดข้อมูลไม่ตรงตามข้อกำหนด
        raise ValueError("Need at least 50 test images and disjoint splits.")
    # สร้างโฟลเดอร์ผลลัพธ์และโฟลเดอร์ examples รวมถึงโฟลเดอร์แม่ที่ขาดหาย
    (out_dir / "examples").mkdir(parents=True, exist_ok=True)

    # เลือก Threshold ตามค่าคงที่หรือจากการจูนบนชุด Validation
    if args.fixed_threshold is not None:
        threshold_before = float(args.fixed_threshold)
        threshold_after = float(args.fixed_threshold)
        tuning = [{"threshold": threshold_before, "mean_iou_before": None, "mean_iou_after": None}]
        print(f"Using fixed threshold: {threshold_before}", flush=True)
    else:
        # เลือก Threshold ที่ดีที่สุดจากชุด Validation ทั้งสำหรับ Before และ After Morphology
        threshold_before, threshold_after, tuning = choose_threshold(
            split["validation"],
            data_dir=data_dir,
            max_image_side=args.max_image_side,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
            distance_scale=args.distance_scale,
            kernel_size=args.kernel_size,
        )
        print(f"Validation selected thresholds -> Before: {threshold_before}, After: {threshold_after}", flush=True)

    # -------------------------------------------------------------------------
    # ส่วนที่ 6.1: การประเมินภาพ Test ก่อนและหลังทำ Morphology
    # -------------------------------------------------------------------------
    totals = {stage: dict(TN=0, FP=0, FN=0, TP=0) for stage in ("before", "after")}
    rows, all_truth, all_scores = [], [], []
    test_samples = []

    for index, name in enumerate(split["test"]):
        rgb, truth, valid = load_sample(name, data_dir=data_dir, max_image_side=args.max_image_side)
        score = foreground_score(rgb, distance_scale=args.distance_scale)
        test_samples.append((rgb, truth, valid, score))

        # Binary mask ก่อนและหลังทำ Morphology โดยใช้ Threshold ที่เหมาะสมของแต่ละขั้นตอน
        raw = score >= threshold_before
        cleaned = clean_mask(score >= threshold_after, kernel_size=args.kernel_size)

        for stage, prediction in [("before", raw), ("after", cleaned)]:
            counts = confusion_counts(truth[valid], prediction[valid])
            for key, count in counts.items():
                totals[stage][key] += count
            rows.append({"image": name, "stage": stage, **counts, **metrics(counts)})

        all_truth.append(truth[valid])
        all_scores.append(score[valid])
        save_example(name, rgb, truth, valid, raw, cleaned, output_dir=out_dir)
        print(f"Evaluated {index + 1}/{len(split['test'])}: {name}", flush=True)

    # -------------------------------------------------------------------------
    # ส่วนที่ 6.2: การคำนวณ ROC Curves (Line 1: Threshold vs Line 2: Morphology)
    # -------------------------------------------------------------------------
    y_true, y_score = np.concatenate(all_truth), np.concatenate(all_scores)

    # Line 1: Threshold ROC Curve (คะแนนดิบต่อเนื่องจากช่อง a*)
    fpr_thresh, tpr_thresh, _ = roc_curve(y_true, y_score)
    auc_thresh = float(roc_auc_score(y_true, y_score))

    # Line 2: Morphology Pipeline ROC Curve (กวาด 99 Thresholds พร้อมรัน clean_mask)
    morph_sweep_thresholds = np.linspace(0.01, 0.99, 99)
    morph_pts = []
    for t in morph_sweep_thresholds:
        t_fp, t_tn, t_tp, t_fn = 0, 0, 0, 0
        for _, truth_img, valid_img, score_img in test_samples:
            cl_m = clean_mask(score_img >= t, kernel_size=args.kernel_size)
            v_truth = truth_img[valid_img]
            v_pred = cl_m[valid_img]
            t_tp += int(np.sum(v_truth & v_pred))
            t_tn += int(np.sum(~v_truth & ~v_pred))
            t_fp += int(np.sum(~v_truth & v_pred))
            t_fn += int(np.sum(v_truth & ~v_pred))
        fpr_v = t_fp / (t_fp + t_tn) if (t_fp + t_tn) > 0 else 0.0
        tpr_v = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0.0
        morph_pts.append((fpr_v, tpr_v, float(t)))

    all_morph_pts = [(0.0, 0.0, 1.0)] + morph_pts + [(1.0, 1.0, 0.0)]
    all_morph_pts.sort(key=lambda x: (x[0], x[1]))
    fpr_morph = np.array([p[0] for p in all_morph_pts])
    tpr_morph = np.array([p[1] for p in all_morph_pts])
    auc_morph = float(sk_auc(fpr_morph, tpr_morph))

    # จุดตัวอย่างบนเส้นทั้งสองเส้นทุกๆ 0.02 FPR (0.00, 0.02, 0.04, ..., 1.00)
    target_fprs = np.arange(0.00, 1.01, 0.02)
    dots_thresh_tpr = np.interp(target_fprs, fpr_thresh, tpr_thresh)
    dots_morph_tpr = np.interp(target_fprs, fpr_morph, tpr_morph)

    # 2 Best Operating Dots
    fpr_best_thresh = totals["before"]["FP"] / (totals["before"]["FP"] + totals["before"]["TN"])
    tpr_best_thresh = totals["before"]["TP"] / (totals["before"]["TP"] + totals["before"]["FN"])

    fpr_best_morph = totals["after"]["FP"] / (totals["after"]["FP"] + totals["after"]["TN"])
    tpr_best_morph = totals["after"]["TP"] / (totals["after"]["TP"] + totals["after"]["FN"])

    summary = {
        "validation_images": len(split["validation"]),
        "test_images": len(split["test"]),
        "threshold_before": threshold_before,
        "threshold_after": threshold_after,
        "max_image_side": args.max_image_side,
        "kernel_size": args.kernel_size,
        "distance_scale": args.distance_scale,
        "ignored_trimap_label": 3,
        "evaluated_pixels": int(y_true.size),
        "auc_threshold_line": auc_thresh,
        "auc_morphology_line": auc_morph,
        "best_dot_threshold": {
            "threshold": threshold_before,
            "fpr": float(fpr_best_thresh),
            "tpr": float(tpr_best_thresh)
        },
        "best_dot_morphology": {
            "threshold": threshold_after,
            "fpr": float(fpr_best_morph),
            "tpr": float(tpr_best_morph)
        },
        "line1_dots_every_002_fpr": [
            {"fpr": round(float(f), 3), "tpr": round(float(t), 4)}
            for f, t in zip(target_fprs, dots_thresh_tpr)
        ],
        "line2_dots_every_002_fpr": [
            {"fpr": round(float(f), 3), "tpr": round(float(t), 4)}
            for f, t in zip(target_fprs, dots_morph_tpr)
        ],
        **{stage: {**counts, **metrics(counts)} for stage, counts in totals.items()},
    }
    if args.fixed_threshold is not None:
        summary["fixed_threshold"] = args.fixed_threshold

    # -------------------------------------------------------------------------
    # ส่วนที่ 6.3: การบันทึกผลลัพธ์เป็น JSON และ CSV
    # -------------------------------------------------------------------------
    for filename, content in [("summary.json", summary), ("threshold_search.json", tuning), ("split.json", split)]:
        write_commented_json(out_dir / filename, content)

    with (out_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # -------------------------------------------------------------------------
    # ส่วนที่ 6.4: การสร้างกราฟ Confusion matrix (ระบุ TP, TN, FP, FN ชัดเจน)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    labels_map = [
        [("TN", "True Negative\n(Background → Background)"), ("FP", "False Positive\n(Background → Item)")],
        [("FN", "False Negative\n(Item → Background)"), ("TP", "True Positive\n(Item → Item)")],
    ]
    for ax, (stage, counts) in zip(axes, totals.items()):
        matrix = np.array([[counts["TN"], counts["FP"]], [counts["FN"], counts["TP"]]])
        ax.imshow(matrix, cmap="Blues")
        for (row, col), value in np.ndenumerate(matrix):
            abbr, desc = labels_map[row][col]
            text_color = "white" if value > matrix.max() * 0.45 else "black"
            ax.text(col, row - 0.13, abbr, ha="center", va="center", color=text_color, fontsize=15, fontweight="bold")
            ax.text(col, row + 0.07, f"{value:,}", ha="center", va="center", color=text_color, fontsize=12, fontweight="medium")
            sub_desc = desc.splitlines()[1].strip("()")
            ax.text(col, row + 0.23, f"({sub_desc})", ha="center", va="center", color=text_color, fontsize=8.5, alpha=0.9)
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Background\n(Predicted)", "Item (Apple)\n(Predicted)"],
            yticklabels=["Background\n(Actual)", "Item (Apple)\n(Actual)"],
            xlabel="Predicted Class",
            ylabel="Actual Class",
            title=f"Stage: {stage.title()}"
        )
    fig.suptitle("Pixel-Level Confusion Matrix: Red Apples vs Textured Surface (CIELAB a*)", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # ส่วนที่ 6.5: การสร้าง Dual ROC Curves พร้อม 2 Best Dots และ Dots ทุก 0.02 FPR
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    # Line 1: Threshold ROC curve
    ax.plot(fpr_thresh, tpr_thresh, color="#1f77b4", linewidth=2.4, label=f"Line 1: Threshold Curve (Raw a* Score, AUC = {auc_thresh:.3f})")
    # Line 2: Morphology ROC curve
    ax.plot(fpr_morph, tpr_morph, color="#2ca02c", linewidth=2.4, label=f"Line 2: Morphology Curve (Opening & Closing, AUC = {auc_morph:.3f})")
    # Random guess baseline
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.5, label="Random Guess Baseline (AUC = 0.500)")

    # Sampled Dots every 0.02 FPR
    ax.scatter(target_fprs, dots_thresh_tpr, color="#1f77b4", s=25, alpha=0.85, zorder=4, label="Line 1 Dots (Every 0.02 FPR)")
    ax.scatter(target_fprs, dots_morph_tpr, color="#2ca02c", s=25, alpha=0.85, zorder=4, label="Line 2 Dots (Every 0.02 FPR)")

    # 2 Best Operating Dots
    ax.scatter([fpr_best_thresh], [tpr_best_thresh], color="#ff7f0e", marker="o", s=130, edgecolors="black", linewidths=1.5, zorder=6, label=f"Best Dot Line 1 (Threshold = {threshold_before:.2f})")
    ax.scatter([fpr_best_morph], [tpr_best_morph], color="#e11d48", marker="o", s=130, edgecolors="black", linewidths=1.5, zorder=6, label=f"Best Dot Line 2 (Morphology = {threshold_after:.2f})")

    # Annotations for 2 Best Dots
    ax.annotate(
        f"Best Morphology Dot\n(t={threshold_after:.2f}, FPR={fpr_best_morph:.3f}, TPR={tpr_best_morph:.3f})",
        xy=(fpr_best_morph, tpr_best_morph),
        xytext=(0.10, 0.88),
        arrowprops=dict(facecolor="#e11d48", edgecolor="#e11d48", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="#ffebee", ec="#e11d48", lw=1.3),
        zorder=7
    )
    ax.annotate(
        f"Best Threshold Dot\n(t={threshold_before:.2f}, FPR={fpr_best_thresh:.3f}, TPR={tpr_best_thresh:.3f})",
        xy=(fpr_best_thresh, tpr_best_thresh),
        xytext=(0.28, 0.65),
        arrowprops=dict(facecolor="#ff7f0e", edgecolor="#ff7f0e", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e1", ec="#ff7f0e", lw=1.3),
        zorder=7
    )

    ax.set(
        xlabel="False Positive Rate (FPR = FP / [FP + TN])",
        ylabel="True Positive Rate (TPR / Recall = TP / [TP + FN])",
        title="Test Pixels: Dual ROC Curves & 0.02 FPR Operating Points (Red Apples Dataset)",
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.05)
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9.0)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # ส่วนที่ 6.6: การสร้างรายงานสรุปผลรูปแบบ Markdown
    # -------------------------------------------------------------------------
    # เริ่มสร้าง List ของข้อความสำหรับรายงานผลรูปแบบ Markdown
    lines = [
        # เพิ่มหัวข้อหลักของรายงานผล
        "# Experiment results: Red Apples on Textured Surfaces (CIELAB a*)",
        "",
        # เพิ่มข้อความระบุคำสั่งและแหล่งข้อมูลที่ใช้สร้างผล
        "Generated by `python run.py` from Red Apple images (5 varieties) on textured surfaces with cast shadows.",
        "",
        # เพิ่มจำนวนภาพ Validation จำนวนภาพ Test และ Threshold
        f"Validation: {len(split['validation'])} images. Test: {len(split['test'])} images. Thresholds: Before={threshold_before}, After={threshold_after}.",
        "",
        # เพิ่มคำอธิบายวิธีรวมพิกเซลและ Label ที่ละเว้น
        "Metrics pool all valid test pixels after resizing; trimap label 3 is excluded.",
        "",
        # เพิ่มชื่อคอลัมน์ของตาราง Metrics
        "| Stage | Accuracy | Precision | Recall | IoU | Dice |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    # วนขั้นตอนก่อนและหลัง Morphology ตามลำดับในยอดรวม
    for stage in totals:
        values = metrics(totals[stage])
        lines.append("| " + stage + " | " + " | ".join(f"{v:.4f}" for v in values.values()) + " |")
    lines += [
        "",
        f"Line 1 (Threshold) ROC AUC: {auc_thresh:.4f}.",
        f"Line 2 (Morphology) ROC AUC: {auc_morph:.4f}.",
        "",
        "Dual ROC curves: Line 1 sweeps threshold on raw a* scores; Line 2 applies morphological opening & closing at every threshold.",
        "",
        "This baseline detects red color using the CIELAB a* channel exclusively. Morphology removes small shadow/grain noise and fills specular reflection gaps.",
    ]
    (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # แสดงข้อมูลสรุปเป็น JSON แบบเยื้องบนหน้าจอ
    print(json.dumps(summary, indent=2))


# =============================================================================
# ส่วนที่ 7: จุดเริ่มต้นการทำงานเมื่อรันไฟล์โดยตรง
# =============================================================================
# ตรวจสอบว่าไฟล์นี้ถูกรันโดยตรงแทนการถูก Import เป็นโมดูล
if __name__ == "__main__":
    # เรียกฟังก์ชันหลักเพื่อเริ่มการทดลองทั้งหมด
    main()
