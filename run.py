# =============================================================================
# ส่วนที่ 1: คำอธิบายโมดูลและไลบรารีที่ใช้
# =============================================================================
# ระบุว่าโมดูลนี้ควบคุมขั้นตอนการทดลองตั้งแต่ปรับ Threshold จนถึงบันทึกผลลัพธ์
"""รันการทดลองทั้งหมด: ปรับ threshold -> ทดสอบ -> บันทึกผลและกราฟ."""
# นำเข้า Path สำหรับสร้างและจัดการพาธไฟล์
from pathlib import Path
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
# ส่วนที่ 3: การโหลดและเตรียมภาพกับ Ground truth
# =============================================================================
# ประกาศฟังก์ชันสำหรับโหลดรูปภาพและ Ground truth หนึ่งตัวอย่าง
def load_sample(name):
    # อธิบายว่าฟังก์ชันย่อภาพและรักษาค่า Label ของ Trimap
    """ย่อภาพให้ด้านยาวไม่เกิน 256 และอ่าน trimap โดยรักษาค่า label."""
    # เปิดไฟล์ภาพสีตามชื่อที่รับเข้ามาและปิดไฟล์อัตโนมัติเมื่ออ่านเสร็จ
    with Image.open(DATA / "images" / (name + ".jpg")) as image:
        # แปลงภาพเป็น RGB แล้วเปลี่ยนเป็นอาร์เรย์ NumPy
        rgb = np.array(image.convert("RGB"))
    # เปิดไฟล์ Trimap ตามชื่อภาพและปิดไฟล์อัตโนมัติเมื่ออ่านเสร็จ
    with Image.open(DATA / "masks" / (name + ".png")) as image:
        # แปลง Trimap เป็นอาร์เรย์ NumPy โดยคงค่า Label เดิม
        trimap = np.array(image)
    # ตรวจสอบว่าขนาด Trimap ตรงกับภาพและมีเฉพาะ Label 1, 2 และ 3
    if trimap.shape != rgb.shape[:2] or not set(np.unique(trimap)) <= {1, 2, 3}:
        # แจ้งข้อผิดพลาดพร้อมชื่อไฟล์เมื่อ Trimap ไม่ถูกต้อง
        raise ValueError(f"Invalid trimap: {name}")
    # อ่านความสูงและความกว้างของภาพจากสองมิติแรก
    height, width = rgb.shape[:2]
    # คำนวณอัตราย่อโดยให้ด้านยาวไม่เกิน 256 พิกเซลและไม่ขยายภาพเล็ก
    scale = min(1.0, 256 / max(height, width))
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
def choose_threshold(names):
    # อธิบายว่าฟังก์ชันเลือก Threshold ด้วยค่าเฉลี่ย IoU ก่อนทำ Morphology
    """เลือก threshold จาก validation เท่านั้น โดยใช้ mean IoU ก่อน Morphology."""
    # สร้าง List ว่างสำหรับเก็บคะแนน Ground truth และ Valid mask ของทุกภาพ
    samples = []
    # วนอ่านภาพ Validation แต่ละชื่อ
    for name in names:
        # โหลดภาพ Ground truth และบริเวณที่ใช้ประเมินของตัวอย่างปัจจุบัน
        rgb, truth, valid = load_sample(name)
        # คำนวณคะแนนวัตถุแล้วเก็บพร้อม Ground truth และ Valid mask
        samples.append((foreground_score(rgb), truth, valid))
    # สร้าง List ว่างสำหรับเก็บผลของ Threshold แต่ละค่า
    candidates = []
    # วน Threshold ตั้งแต่ 0.10 ถึง 0.90 โดยเพิ่มครั้งละ 0.05
    for threshold in np.arange(0.10, 0.91, 0.05):
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
def save_example(name, rgb, truth, valid, raw, cleaned):
    # อธิบายองค์ประกอบทั้งห้าที่จะแสดงในภาพเปรียบเทียบ
    """แสดงภาพจริง เฉลย และ mask ก่อน/หลัง โดยระบายขอบที่ไม่ประเมินเป็นสีเทา."""
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
    fig.savefig(OUT / "examples" / (name + ".png"), dpi=120)
    # ปิด Figure เพื่อคืนหน่วยความจำหลังบันทึกไฟล์
    plt.close(fig)


# =============================================================================
# ส่วนที่ 6: กระบวนการทดลองหลักและการตรวจสอบชุดข้อมูล
# =============================================================================
# ประกาศฟังก์ชันหลักที่ควบคุมการทดลองทั้งหมด
def main():
    # ตรวจสอบว่ามีไฟล์แบ่งชุดข้อมูลที่สร้างโดย download_data.py แล้วหรือไม่
    if not (DATA / "split.json").exists():
        # หยุดโปรแกรมพร้อมแจ้งให้ดาวน์โหลดข้อมูลก่อนเมื่อไม่พบไฟล์ Split
        raise SystemExit("Run python download_data.py first.")
    # อ่านข้อความ JSON ของ Split แล้วแปลงเป็น Dictionary
    split = json.loads((DATA / "split.json").read_text(encoding="utf-8"))
    # ตรวจสอบว่ามีภาพ Test อย่างน้อย 50 ภาพและไม่มีภาพซ้ำกับ Validation
    if len(split["test"]) < 50 or set(split["validation"]) & set(split["test"]):
        # แจ้งข้อผิดพลาดเมื่อขนาดหรือการแยกชุดข้อมูลไม่ตรงตามข้อกำหนด
        raise ValueError("Need at least 50 test images and disjoint splits.")
    # สร้างโฟลเดอร์ผลลัพธ์และโฟลเดอร์ examples รวมถึงโฟลเดอร์แม่ที่ขาดหาย
    (OUT / "examples").mkdir(parents=True, exist_ok=True)
    # เลือก Threshold ที่ดีที่สุดจากชุด Validation และเก็บผลค้นหาทุกค่า
    threshold, tuning = choose_threshold(split["validation"])
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
        rgb, truth, valid = load_sample(name)
        # คำนวณคะแนนความเป็นวัตถุของทุกพิกเซล
        score = foreground_score(rgb)
        # เปรียบเทียบคะแนนกับ Threshold เพื่อสร้าง Binary mask ก่อน Morphology
        raw = score >= threshold
        # ทำ Opening และ Closing เพื่อสร้าง Binary mask หลัง Morphology
        cleaned = clean_mask(raw)
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
        save_example(name, rgb, truth, valid, raw, cleaned)
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
        "max_image_side": 256,
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

    # -------------------------------------------------------------------------
    # ส่วนที่ 6.3: การบันทึกผลลัพธ์เป็น JSON และ CSV
    # -------------------------------------------------------------------------
    # วนชื่อไฟล์และข้อมูลสำหรับผลลัพธ์ JSON ทั้งสามไฟล์
    for filename, content in [("summary.json", summary), ("threshold_search.json", tuning), ("split.json", split)]:
        # แปลงข้อมูลเป็น JSON แบบเยื้องแล้วบันทึกด้วย UTF-8
        write_commented_json(OUT / filename, content)
    # เปิดไฟล์ CSV ผลรายภาพในโหมดเขียนโดยไม่เพิ่มบรรทัดว่างซ้ำ
    with (OUT / "per_image.csv").open("w", newline="", encoding="utf-8") as file:
        # สร้างตัวเขียน CSV โดยใช้ Key ของผลลัพธ์แถวแรกเป็นชื่อคอลัมน์
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        # เขียนแถวชื่อคอลัมน์ลงไฟล์ CSV
        writer.writeheader()
        # เขียนผลลัพธ์ของทุกภาพและทุกขั้นตอนลงไฟล์ CSV
        writer.writerows(rows)

    # -------------------------------------------------------------------------
    # ส่วนที่ 6.4: การสร้างกราฟ Confusion matrix
    # -------------------------------------------------------------------------
    # สร้าง Figure สำหรับ Confusion matrix ก่อนและหลัง Morphology สองช่อง
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    # วนจับคู่ Axes กับชื่อขั้นตอนและยอดรวม Confusion matrix
    for ax, (stage, counts) in zip(axes, totals.items()):
        # จัด TN, FP, FN และ TP เป็นเมทริกซ์ขนาด 2 คูณ 2
        matrix = np.array([[counts["TN"], counts["FP"]], [counts["FN"], counts["TP"]]])
        # แสดงเมทริกซ์เป็นภาพ Heatmap โทนสีน้ำเงิน
        ax.imshow(matrix, cmap="Blues")
        # วนตำแหน่งแถว คอลัมน์ และค่าของทุกช่องในเมทริกซ์
        for (row, col), value in np.ndenumerate(matrix):
            # เขียนจำนวนพิกเซลกลางช่องและเลือกสีข้อความตามความเข้มของพื้นหลัง
            ax.text(col, row, f"{value:,}", ha="center", va="center", color="white" if value > matrix.max() / 2 else "black")
        # กำหนด Tick ชื่อคลาส ชื่อแกน และหัวข้อของ Confusion matrix ปัจจุบัน
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Background", "Pet"], yticklabels=["Background", "Pet"], xlabel="Predicted", ylabel="Actual", title=stage.title())
    # ปรับระยะห่างขององค์ประกอบใน Figure Confusion matrix
    fig.tight_layout()
    # บันทึก Figure Confusion matrix ด้วยความละเอียด 150 DPI
    fig.savefig(OUT / "confusion_matrix.png", dpi=150)
    # ปิด Figure Confusion matrix เพื่อคืนหน่วยความจำ
    plt.close(fig)

    # -------------------------------------------------------------------------
    # ส่วนที่ 6.5: การสร้าง ROC curve และจุดทำงานของ Binary mask
    # -------------------------------------------------------------------------
    # สร้าง Figure และ Axes สำหรับ ROC curve
    fig, ax = plt.subplots(figsize=(6, 5))
    # วาด ROC curve ของคะแนนความต่างสีพร้อมแสดงค่า AUC ในคำอธิบายเส้น
    ax.plot(fpr, tpr, label=f"Color-distance score (AUC={auc:.3f})")
    # วาดเส้นทแยงอ้างอิงของตัวจำแนกแบบสุ่ม
    ax.plot([0, 1], [0, 1], "--", color="gray")
    # อธิบายว่า Binary mask หลัง Morphology แสดงได้เป็นจุดทำงาน ไม่ใช่ ROC curve ต่อเนื่อง
    # Morphology ให้ binary mask จึงแสดงเป็นจุดทำงาน ไม่สร้าง ROC จาก mask 0/1
    # วนข้อมูลก่อนและหลัง Morphology เพื่อเพิ่มจุดทำงานลงกราฟ
    for stage, counts in totals.items():
        # คำนวณ False positive rate จาก FP หารด้วยพิกเซลพื้นหลังจริงทั้งหมด
        fp_rate = counts["FP"] / (counts["FP"] + counts["TN"])
        # วาดจุดทำงานด้วย False positive rate และ Recall ของขั้นตอนปัจจุบัน
        ax.scatter(fp_rate, metrics(counts)["recall"], label=stage.title(), s=45)
    # กำหนดชื่อแกน หัวข้อ และช่วงค่าของ ROC curve
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Test pixels: ROC", xlim=(0, 1), ylim=(0, 1))
    # แสดงคำอธิบายเส้นและจุดที่มุมขวาล่างของกราฟ
    ax.legend(loc="lower right")
    # ปรับระยะห่างขององค์ประกอบใน Figure ROC curve
    fig.tight_layout()
    # บันทึก Figure ROC curve ด้วยความละเอียด 150 DPI
    fig.savefig(OUT / "roc_curve.png", dpi=150)
    # ปิด Figure ROC curve เพื่อคืนหน่วยความจำ
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
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # แสดงข้อมูลสรุปเป็น JSON แบบเยื้องบนหน้าจอ
    print(json.dumps(summary, indent=2))


# =============================================================================
# ส่วนที่ 7: จุดเริ่มต้นการทำงานเมื่อรันไฟล์โดยตรง
# =============================================================================
# ตรวจสอบว่าไฟล์นี้ถูกรันโดยตรงแทนการถูก Import เป็นโมดูล
if __name__ == "__main__":
    # เรียกฟังก์ชันหลักเพื่อเริ่มการทดลองทั้งหมด
    main()
