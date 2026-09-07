"""อัลกอริทึมพื้นฐาน: สีต่างจากขอบภาพมาก -> มีแนวโน้มเป็นวัตถุ."""
import cv2
import numpy as np


def foreground_score(rgb):
    """รับภาพ RGB คืนคะแนน 0..1 ต่อพิกเซล โดยไม่ใช้ Ground truth."""
    # Lab แยกความสว่าง (L) และองค์ประกอบสี (a, b)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    # สมมติว่าขอบภาพส่วนใหญ่เป็นพื้นหลัง ใช้ median ลดผลของสีสุดโต่ง
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    background = np.median(border, axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    # คะแนนเป็นระยะห่างที่ปรับสเกล ไม่ใช่ probability ที่ผ่านการ calibration
    return distance / (distance + 50.0)


def clean_mask(mask):
    """Opening ลบจุดเล็ก; Closing เติมช่องว่างเล็ก ด้วย kernel 3x3."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel).astype(bool)


def confusion_counts(truth, prediction):
    """นับผลระดับพิกเซล; 1=สัตว์, 0=พื้นหลัง."""
    return {
        "TN": int(np.sum(~truth & ~prediction)),
        "FP": int(np.sum(~truth & prediction)),
        "FN": int(np.sum(truth & ~prediction)),
        "TP": int(np.sum(truth & prediction)),
    }


def metrics(counts):
    tn, fp, fn, tp = (counts[key] for key in ("TN", "FP", "FN", "TP"))
    def divide(a, b):
        return a / b if b else 0.0
    return {
        "accuracy": divide(tp + tn, tp + tn + fp + fn),
        "precision": divide(tp, tp + fp),
        "recall": divide(tp, tp + fn),
        "iou": divide(tp, tp + fp + fn),
        "dice": divide(2 * tp, 2 * tp + fp + fn),
    }
