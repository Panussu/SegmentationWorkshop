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
from sklearn.metrics import roc_curve, roc_auc_score

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
    parser.add_argument("--threshold-min", type=float, default=0.10, help="ค่า Threshold ต่ำสุดในการค้นหา")
    parser.add_argument("--threshold-max", type=float, default=0.90, help="ค่า Threshold สูงสุดในการค้นหา")
    parser.add_argument("--threshold-step", type=float, default=0.05, help="สเต็ปในการค้นหา Threshold")
    parser.add_argument("--fixed-threshold", type=float, default=None, help="ระบุ Threshold คงที่โดยตรง ข้ามขั้นตอนจูนบน Validation")
    parser.add_argument("--kernel-size", type=int, default=3, help="ขนาด Kernel สำหรับ Morphology ใน clean_mask (เลขคี่)")
    parser.add_argument("--distance-scale", type=float, default=50.0, help="ตัวหารปรับสเกลคะแนนระยะสีใน foreground_score")

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
    # 1=สัตว์, 2=พื้นหลัง, 3=บริเวณขอบที่ไม่แน่นอน (ไม่นำมาคิดคะแนน)
    # คืนภาพ RGB, Mask วัตถุ และ Mask ระบุพิกเซลที่ใช้ประเมินผล
    return rgb, trimap == 1, trimap != 3


# =============================================================================
# ส่วนที่ 4: การค้นหา Threshold ที่เหมาะสมจากชุด Validation
# =============================================================================
# ประกาศฟังก์ชันสำหรับเลือก Threshold จากชุด Validation
def choose_threshold(names, data_dir=DATA, max_image_side=256, threshold_min=0.10, threshold_max=0.90, threshold_step=0.05, distance_scale=50.0):
    # อธิบายว่าฟังก์ชันเลือก Threshold ด้วยค่าเฉลี่ย IoU ก่อนทำ Morphology
    """เลือก threshold จาก validation เท่านั้น โดยใช้ mean IoU ก่อน Morphology."""
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
        # คำนวณ IoU ของทุกภาพด้วย Threshold ปัจจุบัน โดยใช้เฉพาะพิกเซลที่ Valid
        scores = [metrics(confusion_counts(truth[valid], (score >= threshold)[valid]))["iou"] for score, truth, valid in samples]
        # เก็บค่า Threshold ที่ปัดเป็นสองตำแหน่งและค่าเฉลี่ย IoU ของทุกภาพ
        candidates.append({"threshold": round(float(threshold), 2), "mean_iou": float(np.mean(scores))})
    # เลือก Threshold ที่มี Mean IoU สูงสุดและคืนผลการค้นหาทั้งหมดด้วย
    return max(candidates, key=lambda item: item["mean_iou"])["threshold"], candidates


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
        threshold = float(args.fixed_threshold)
        tuning = [{"threshold": threshold, "mean_iou": None}]
        print(f"Using fixed threshold: {threshold}", flush=True)
    else:
        # เลือก Threshold ที่ดีที่สุดจากชุด Validation และเก็บผลค้นหาทุกค่า
        threshold, tuning = choose_threshold(
            split["validation"],
            data_dir=data_dir,
            max_image_side=args.max_image_side,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
            distance_scale=args.distance_scale,
        )
        # แสดง Threshold ที่เลือกและบังคับให้ข้อความแสดงทันที
        print(f"Validation selected threshold: {threshold}", flush=True)

    # -------------------------------------------------------------------------
    # ส่วนที่ 6.1: การประเมินภาพ Test ก่อนและหลังทำ Morphology
    # -------------------------------------------------------------------------
    # สร้างยอดรวม Confusion matrix แยกก่อนและหลังทำ Morphology
    totals = {stage: dict(TN=0, FP=0, FN=0, TP=0) for stage in ("before", "after")}
    # สร้าง List สำหรับผลรายภาพ Ground truth รวม และคะแนนรวมตามลำดับ
    rows, all_truth, all_scores = [], [], []
    # วนทดสอบภาพทุกชื่อพร้อมเลขลำดับของภาพ
    for index, name in enumerate(split["test"]):
        # โหลดภาพ Ground truth และบริเวณที่ใช้ประเมินของภาพปัจจุบัน
        rgb, truth, valid = load_sample(name, data_dir=data_dir, max_image_side=args.max_image_side)
        # คำนวณคะแนนความเป็นวัตถุของทุกพิกเซล
        score = foreground_score(rgb, distance_scale=args.distance_scale)
        # เปรียบเทียบคะแนนกับ Threshold เพื่อสร้าง Binary mask ก่อน Morphology
        raw = score >= threshold
        # ทำ Opening และ Closing เพื่อสร้าง Binary mask หลัง Morphology
        cleaned = clean_mask(raw, kernel_size=args.kernel_size)
        # วนประเมินผลทั้ง Mask ก่อนและหลังทำ Morphology
        for stage, prediction in [("before", raw), ("after", cleaned)]:
            # นับ TN, FP, FN และ TP จากพิกเซลที่ Valid เท่านั้น
            counts = confusion_counts(truth[valid], prediction[valid])
            # วนค่าทุกช่องของ Confusion matrix ในภาพปัจจุบัน
            for key, count in counts.items():
                # บวกจำนวนพิกเซลเข้ากับยอดรวมของขั้นตอนปัจจุบัน
                totals[stage][key] += count
            # เก็บชื่อภาพ ขั้นตอน จำนวน Confusion matrix และ Metrics ลงผลรายภาพ
            rows.append({"image": name, "stage": stage, **counts, **metrics(counts)})
        # เก็บ Ground truth ของพิกเซลที่ Valid สำหรับคำนวณ ROC รวม
        all_truth.append(truth[valid])
        # เก็บคะแนนของพิกเซลที่ Valid สำหรับคำนวณ ROC รวม
        all_scores.append(score[valid])
        # อธิบายว่าบันทึกตัวอย่างทุกภาพโดยไม่คัดเลือกเฉพาะภาพที่ให้ผลดี
        # เก็บตัวอย่างทุกภาพ ไม่เลือกเฉพาะภาพที่ผลดี
        # บันทึกภาพเปรียบเทียบของตัวอย่างปัจจุบัน
        save_example(name, rgb, truth, valid, raw, cleaned, output_dir=out_dir)
        # แสดงความคืบหน้าพร้อมลำดับและชื่อภาพที่ประเมินเสร็จ
        print(f"Evaluated {index + 1}/{len(split['test'])}: {name}", flush=True)

    # -------------------------------------------------------------------------
    # ส่วนที่ 6.2: การคำนวณ ROC AUC และสร้างข้อมูลสรุป
    # -------------------------------------------------------------------------
    # รวม Ground truth และคะแนนของทุกภาพเป็นอาร์เรย์หนึ่งมิติขนาดใหญ่
    y_true, y_score = np.concatenate(all_truth), np.concatenate(all_scores)
    # คำนวณ False positive rate และ True positive rate ของ ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # คำนวณพื้นที่ใต้ ROC curve แล้วแปลงเป็น float มาตรฐาน
    auc = float(roc_auc_score(y_true, y_score))
    # เริ่มสร้าง Dictionary สรุปข้อมูลการทดลองและผลประเมิน
    summary = {
        # บันทึกจำนวนภาพในชุด Validation
        "validation_images": len(split["validation"]),
        # บันทึกจำนวนภาพในชุด Test
        "test_images": len(split["test"]),
        # บันทึก Threshold ที่เลือกจากชุด Validation
        "threshold": threshold,
        # บันทึกขนาดด้านยาวสูงสุดที่ใช้ย่อภาพ
        "max_image_side": args.max_image_side,
        # บันทึกขนาด Kernel ของ Morphology
        "kernel_size": args.kernel_size,
        # บันทึกตัวหารปรับสเกลระยะห่างสี
        "distance_scale": args.distance_scale,
        # บันทึก Label ของ Trimap ที่ไม่ถูกนำมาประเมิน
        "ignored_trimap_label": 3,
        # บันทึกจำนวนพิกเซลที่ถูกนำมาประเมินทั้งหมด
        "evaluated_pixels": int(y_true.size),
        # บันทึกค่า ROC AUC ของคะแนนต่อเนื่อง
        "score_roc_auc": auc,
        # เพิ่มจำนวน Confusion matrix และ Metrics แยกตามขั้นตอนก่อนและหลัง Morphology
        **{stage: {**counts, **metrics(counts)} for stage, counts in totals.items()},
    # ปิด Dictionary สรุปผลการทดลอง
    }
    if args.fixed_threshold is not None:
        summary["fixed_threshold"] = args.fixed_threshold

    # -------------------------------------------------------------------------
    # ส่วนที่ 6.3: การบันทึกผลลัพธ์เป็น JSON และ CSV
    # -------------------------------------------------------------------------
    # วนชื่อไฟล์และข้อมูลสำหรับผลลัพธ์ JSON ทั้งสามไฟล์
    for filename, content in [("summary.json", summary), ("threshold_search.json", tuning), ("split.json", split)]:
        # แปลงข้อมูลเป็น JSON แบบเยื้องแล้วบันทึกด้วย UTF-8
        write_commented_json(out_dir / filename, content)
    # เปิดไฟล์ CSV ผลรายภาพในโหมดเขียนโดยไม่เพิ่มบรรทัดว่างซ้ำ
    with (out_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as file:
        # สร้างตัวเขียน CSV โดยใช้ Key ของผลลัพธ์แถวแรกเป็นชื่อคอลัมน์
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        # เขียนแถวชื่อคอลัมน์ลงไฟล์ CSV
        writer.writeheader()
        # เขียนผลลัพธ์ของทุกภาพและทุกขั้นตอนลงไฟล์ CSV
        writer.writerows(rows)

    # -------------------------------------------------------------------------
    # ส่วนที่ 6.4: การสร้างกราฟ Confusion matrix (ระบุ TP, TN, FP, FN ชัดเจน)
    # -------------------------------------------------------------------------
    # สร้าง Figure สำหรับ Confusion matrix ก่อนและหลัง Morphology สองช่อง
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    labels_map = [
        [("TN", "True Negative\n(Background → Background)"), ("FP", "False Positive\n(Background → Pet)")],
        [("FN", "False Negative\n(Pet → Background)"), ("TP", "True Positive\n(Pet → Pet)")],
    ]
    for ax, (stage, counts) in zip(axes, totals.items()):
        # จัด TN, FP, FN และ TP เป็นเมทริกซ์ขนาด 2 คูณ 2
        matrix = np.array([[counts["TN"], counts["FP"]], [counts["FN"], counts["TP"]]])
        ax.imshow(matrix, cmap="Blues")
        for (row, col), value in np.ndenumerate(matrix):
            abbr, desc = labels_map[row][col]
            text_color = "white" if value > matrix.max() * 0.45 else "black"
            # แสดงชื่อย่อ TP, TN, FP, FN ตัวหนาขนาดใหญ่
            ax.text(col, row - 0.13, abbr, ha="center", va="center", color=text_color, fontsize=15, fontweight="bold")
            # แสดงจำนวนพิกเซลจริง
            ax.text(col, row + 0.07, f"{value:,}", ha="center", va="center", color=text_color, fontsize=12, fontweight="medium")
            # แสดงคำอธิบายการทำนายสั้นๆ
            sub_desc = desc.splitlines()[1].strip("()")
            ax.text(col, row + 0.23, f"({sub_desc})", ha="center", va="center", color=text_color, fontsize=8.5, alpha=0.9)
        # กำหนด Tick และชื่อแกนอย่างชัดเจน
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Background\n(Predicted)", "Pet\n(Predicted)"],
            yticklabels=["Background\n(Actual)", "Pet\n(Actual)"],
            xlabel="Predicted Class",
            ylabel="Actual Class",
            title=f"Stage: {stage.title()}"
        )
    fig.suptitle("Pixel-Level Confusion Matrix (TP, TN, FP, FN Breakdown)", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # ส่วนที่ 6.5: การสร้าง ROC curve และจุดทำงานของ Binary mask พร้อมคำอธิบาย
    # -------------------------------------------------------------------------
    # สร้าง Figure และ Axes สำหรับ ROC curve ขนาดใหญ่ชัดเจน
    fig, ax = plt.subplots(figsize=(8, 7.2))
    # วาดเส้น ROC ของคะแนนความต่างสีต่อเนื่อง
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2.3, label=f"Color-distance score curve (AUC = {auc:.3f})")
    # วาดเส้นประอ้างอิงการสุ่มเดา
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.5, label="Random guess baseline (AUC = 0.500)")

    # พล็อตจุดทำงานของ Before และ After
    colors = {"before": "#ff7f0e", "after": "#2ca02c"}
    pts = {}
    for stage, counts in totals.items():
        fp_rate = counts["FP"] / (counts["FP"] + counts["TN"])
        rec = metrics(counts)["recall"]
        pts[stage] = (fp_rate, rec)
        ax.scatter(fp_rate, rec, color=colors[stage], label=f"Operating Point: {stage.title()}", s=110, zorder=5)

    # ใส่คำอธิบายพร้อมลูกศรชี้จุด Before
    ax.annotate(
        f"Before (Threshold={threshold:.2f})\nFPR={pts['before'][0]:.3f}, TPR={pts['before'][1]:.3f}",
        xy=pts["before"],
        xytext=(pts["before"][0] + 0.08, pts["before"][1] - 0.09),
        arrowprops=dict(facecolor="#ff7f0e", edgecolor="#ff7f0e", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e1", ec="#ff7f0e", lw=1.3),
        zorder=6
    )
    # ใส่คำอธิบายพร้อมลูกศรชี้จุด After
    ax.annotate(
        f"After Morphology\nFPR={pts['after'][0]:.3f}, TPR={pts['after'][1]:.3f}\n(FPR drops: noise removed)",
        xy=pts["after"],
        xytext=(pts["after"][0] - 0.38, pts["after"][1] + 0.06),
        arrowprops=dict(facecolor="#2ca02c", edgecolor="#2ca02c", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="#e8f5e9", ec="#2ca02c", lw=1.3),
        zorder=6
    )

    # กล่องข้อความอธิบายว่าทำไมมีเพียงเส้นเดียว
    why_one_line = (
        "WHY ONLY 1 ROC CURVE (NOT TWO)?\n"
        "1. The solid curve requires continuous scores (0.0 to 1.0) to sweep thresholds.\n"
        "2. Morphology operates on a binary mask (0 or 1) with no continuous probabilities.\n"
        "   Thus, Morphology forms a single discrete Operating Point (Dot), not a new curve."
    )
    ax.text(
        0.03, 0.03, why_one_line,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#adb5bd", lw=1.2, alpha=0.95)
    )

    ax.set(
        xlabel="False Positive Rate (FPR = FP / [FP + TN])",
        ylabel="True Positive Rate (TPR / Recall = TP / [TP + FN])",
        title="Test Pixels: ROC Curve & Operating Points",
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.05)
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # ส่วนที่ 6.6: การสร้างรายงานสรุปผลรูปแบบ Markdown
    # -------------------------------------------------------------------------
    # เริ่มสร้าง List ของข้อความสำหรับรายงานผลรูปแบบ Markdown
    lines = [
        # เพิ่มหัวข้อหลักของรายงานผล
        "# Experiment results",
        # เพิ่มบรรทัดว่างหลังหัวข้อหลัก
        "",
        # เพิ่มข้อความระบุคำสั่งและแหล่งข้อมูลที่ใช้สร้างผล
        "Generated by `python run.py` from real Oxford-IIIT Pet images.",
        # เพิ่มบรรทัดว่างก่อนข้อมูลการแบ่งชุด
        "",
        # เพิ่มจำนวนภาพ Validation จำนวนภาพ Test และ Threshold
        f"Validation: {len(split['validation'])} images. Test: {len(split['test'])} images. Threshold: {threshold}.",
        # เพิ่มบรรทัดว่างก่อนคำอธิบาย Metrics
        "",
        # เพิ่มคำอธิบายวิธีรวมพิกเซลและ Label ที่ละเว้น
        "Metrics pool all valid test pixels after resizing; trimap label 3 is excluded.",
        # เพิ่มบรรทัดว่างก่อนตาราง Metrics
        "",
        # เพิ่มชื่อคอลัมน์ของตาราง Metrics
        "| Stage | Accuracy | Precision | Recall | IoU | Dice |",
        # เพิ่มแนวจัดรูปแบบและการจัดชิดของคอลัมน์ตาราง
        "|---|---:|---:|---:|---:|---:|",
    # ปิด List เริ่มต้นของข้อความรายงาน
    ]
    # วนขั้นตอนก่อนและหลัง Morphology ตามลำดับในยอดรวม
    for stage in totals:
        # คำนวณ Metrics รวมของขั้นตอนปัจจุบัน
        values = metrics(totals[stage])
        # จัดรูป Metrics เป็นทศนิยมสี่ตำแหน่งแล้วเพิ่มเป็นหนึ่งแถวในตาราง
        lines.append("| " + stage + " | " + " | ".join(f"{v:.4f}" for v in values.values()) + " |")
    # เพิ่มบรรทัดว่าง ค่า AUC และคำอธิบายข้อจำกัดต่อท้ายรายงาน
    lines += [
        # เพิ่มบรรทัดว่างหลังตาราง Metrics
        "",
        # เพิ่มค่า ROC AUC ของคะแนนต่อเนื่องเป็นทศนิยมสี่ตำแหน่ง
        f"Continuous score ROC AUC: {auc:.4f}.",
        # เพิ่มบรรทัดว่างก่อนคำอธิบาย Morphology
        "",
        # เพิ่มคำอธิบายว่าผล Morphology แสดงเป็นจุดทำงาน
        "Morphology is shown as an operating point, not a separate ROC curve.",
        # เพิ่มบรรทัดว่างก่อนคำอธิบายข้อจำกัด
        "",
        # เพิ่มข้อจำกัดของสมมติฐานพื้นหลังและผลกระทบที่อาจเกิดจาก Morphology
        "This simple baseline assumes the border represents background. Similar pet/background colors and pets touching the border can cause errors. Morphology can remove small details and does not guarantee an improvement.",
    # ปิด List ของข้อความที่เพิ่มต่อท้ายรายงาน
    ]
    # รวมข้อความด้วยอักขระขึ้นบรรทัดใหม่และบันทึกเป็นไฟล์ Markdown แบบ UTF-8
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
