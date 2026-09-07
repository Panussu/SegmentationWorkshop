"""เพิ่มคำอธิบายใน _comment โดยคงชื่อฟิลด์และค่าผลทดลองเดิมไว้."""
# ใช้ json แปลงข้อมูล Python เป็นข้อความ JSON ที่ถูกต้อง
import json

# จับคู่ชื่อฟิลด์กับความหมายสำหรับผู้อ่านไฟล์
COMMENTS = {
    "validation": "รายชื่อภาพสำหรับเลือก threshold ไม่ใช่ชุดทดสอบผลสุดท้าย",
    "test": "รายชื่อภาพสำหรับประเมินผลหลังเลือก threshold แล้ว",
    "threshold": "คะแนนขั้นต่ำที่เลือกพิกเซลเป็นสัตว์: score >= threshold",
    "mean_iou": "ค่าเฉลี่ย IoU รายภาพบน validation ก่อน Morphology ใช้เลือก threshold ที่ดีที่สุด",
    "validation_images": "จำนวนภาพที่ใช้เลือก threshold",
    "test_images": "จำนวนภาพที่ใช้ประเมินผลจริง",
    "max_image_side": "ความยาวด้านที่ยาวที่สุดหลังย่อภาพ หน่วยพิกเซล",
    "ignored_trimap_label": "ค่า label ของขอบที่ไม่แน่นอน ซึ่งไม่นำมาคิดคะแนน",
    "evaluated_pixels": "จำนวนพิกเซล test ที่ประเมินทั้งหมด หลังย่อภาพและตัด label ที่ไม่ประเมินออก",
    "score_roc_auc": "พื้นที่ใต้ ROC จากคะแนนสีต่อเนื่องของพิกเซล test ก่อน Threshold และ Morphology",
    "before": "ผลรวมระดับพิกเซลก่อน Morphology ไม่ใช่ค่าเฉลี่ย metrics รายภาพ",
    "after": "ผลรวมระดับพิกเซลหลัง Opening และ Closing ไม่ใช่ค่าเฉลี่ย metrics รายภาพ",
    "TN": "จำนวนพิกเซลพื้นหลังที่ทำนายถูกว่าเป็นพื้นหลัง",
    "FP": "จำนวนพิกเซลพื้นหลังที่ทำนายผิดว่าเป็นสัตว์",
    "FN": "จำนวนพิกเซลสัตว์ที่ทำนายผิดว่าเป็นพื้นหลัง",
    "TP": "จำนวนพิกเซลสัตว์ที่ทำนายถูกว่าเป็นสัตว์",
    "accuracy": "สัดส่วนพิกเซลที่ทำนายถูก: (TP + TN) / (TP + TN + FP + FN)",
    "precision": "ความถูกต้องของพิกเซลที่เลือกเป็นสัตว์: TP / (TP + FP)",
    "recall": "สัดส่วนสัตว์จริงที่ตรวจพบ: TP / (TP + FN)",
    "iou": "พื้นที่ทับซ้อนหารพื้นที่รวม: TP / (TP + FP + FN)",
    "dice": "คะแนนความซ้อนทับ: 2TP / (2TP + FP + FN)",
}


# รับข้อมูลโดยไม่เปลี่ยน object ต้นฉบับ แล้วคืนสำเนาที่มีคำอธิบาย
def with_comments(value):
    # ถ้าเป็นรายการ เช่น threshold_search ให้ใส่คำอธิบายแต่ละรายการ
    if isinstance(value, list):
        # เรียกซ้ำกับสมาชิกทุกตัว โดยรักษาลำดับเดิม
        return [with_comments(item) for item in value]
    # ถ้าเป็น dictionary ให้เพิ่มคำอธิบายเฉพาะฟิลด์ที่มีอยู่จริง
    if isinstance(value, dict):
        # ตัดคำอธิบายเก่าออก เพื่อไม่เพิ่มซ้อนเมื่อเขียนไฟล์ซ้ำ
        data = {key: with_comments(item) for key, item in value.items() if key != "_comment"}
        # วางคำอธิบายไว้ต้น object; **data ตามด้วยข้อมูลจริงทั้งหมด
        return {"_comment": {key: COMMENTS[key] for key in data if key in COMMENTS}, **data}
    # ตัวเลขและข้อความ เช่นชื่อภาพ ไม่ต้องเปลี่ยนค่า
    return value


# บันทึก JSON พร้อมคำอธิบายด้วยรูปแบบที่ Python อ่านกลับได้
def write_commented_json(path, value):
    # ensure_ascii=False ทำให้เห็นภาษาไทยตรง ๆ แทนรหัส Unicode
    path.write_text(json.dumps(with_comments(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
