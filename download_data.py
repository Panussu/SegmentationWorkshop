# =============================================================================
# ส่วนที่ 1: คำอธิบายโมดูลและไลบรารีที่ใช้
# =============================================================================
# ระบุว่าโมดูลนี้ใช้ดาวน์โหลดและเตรียมชุดข้อมูล Oxford-IIIT Pet
"""ดาวน์โหลดภาพและ Ground truth จาก Oxford-IIIT Pet (ทำครั้งแรกครั้งเดียว)."""
# นำเข้า Path สำหรับสร้างและจัดการพาธไฟล์แบบข้ามระบบปฏิบัติการ
from pathlib import Path
# นำเข้า random สำหรับสุ่มรายชื่อภาพด้วย Seed ที่กำหนด
import random
# นำเข้า tarfile สำหรับอ่านไฟล์บีบอัดนามสกุล tar.gz
import tarfile
# นำเข้า urllib.request สำหรับดาวน์โหลดไฟล์ผ่าน URL
import urllib.request
# นำเข้า json สำหรับบันทึกรายชื่อภาพที่แบ่งเป็นแต่ละชุด
import json

# =============================================================================
# ส่วนที่ 2: การกำหนดพาธและแหล่งดาวน์โหลดชุดข้อมูล
# =============================================================================
# หาพาธโฟลเดอร์รากจากตำแหน่งของไฟล์ Python ปัจจุบัน
ROOT = Path(__file__).resolve().parent
# กำหนดพาธโฟลเดอร์ที่ใช้เก็บชุดข้อมูล
DATA = ROOT / "data"
# กำหนด URL หลักของไฟล์ชุดข้อมูล Oxford-IIIT Pet
SOURCE = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/"


# =============================================================================
# ส่วนที่ 3: ฟังก์ชันดาวน์โหลดและจัดเก็บไฟล์ Archive
# =============================================================================
# ประกาศฟังก์ชันสำหรับดาวน์โหลดไฟล์หนึ่งไฟล์จากแหล่งข้อมูล
def download(name):
    # อธิบายว่าฟังก์ชันเก็บไฟล์ Archive เพื่อให้เรียกใช้ซ้ำได้
    """เก็บ archive ไว้ใช้ซ้ำ; เปลี่ยนชื่อเมื่อดาวน์โหลดสำเร็จเท่านั้น."""
    # สร้างพาธปลายทางจากโฟลเดอร์ข้อมูลและชื่อไฟล์
    target = DATA / name
    # ตรวจสอบว่าไฟล์ปลายทางยังไม่มีอยู่ก่อนเริ่มดาวน์โหลด
    if not target.exists():
        # แสดงชื่อไฟล์ที่กำลังดาวน์โหลดและบังคับให้ข้อความแสดงทันที
        print(f"Downloading {name} ...", flush=True)
        # สร้างชื่อไฟล์ชั่วคราวนามสกุล .part สำหรับการดาวน์โหลดที่ยังไม่เสร็จ
        partial = target.with_suffix(".part")
        # ดาวน์โหลดข้อมูลจาก URL ต้นทางลงไฟล์ชั่วคราว
        urllib.request.urlretrieve(SOURCE + name, partial)
        # เปลี่ยนชื่อไฟล์ชั่วคราวเป็นชื่อจริงเมื่อดาวน์โหลดสำเร็จ
        partial.replace(target)
    # ส่งพาธของไฟล์ที่พร้อมใช้งานกลับไป
    return target


# =============================================================================
# ส่วนที่ 4: การเลือก แบ่ง และแตกไฟล์ชุดข้อมูลที่ต้องการใช้งาน
# =============================================================================
# ประกาศฟังก์ชันหลักสำหรับดาวน์โหลดและจัดชุดข้อมูลทั้งหมด
def main():
    # สร้างโฟลเดอร์ข้อมูลหากยังไม่มีอยู่
    DATA.mkdir(exist_ok=True)
    # ดาวน์โหลดไฟล์ Archive ที่เก็บ Annotation และ Trimap
    annotations = download("annotations.tar.gz")
    # อธิบายจำนวนภาพ Validation และ Test ที่ต้องสุ่มจากรายการต้นฉบับ
    # ใช้รายการ trainval เลือก 20 ภาพปรับ threshold และ test เลือก 50 ภาพทดสอบ
    # เปิด Archive ของ Annotation เพื่ออ่านรายชื่อและดึง Mask ที่ต้องการ
    with tarfile.open(annotations) as archive:
        # สร้างตัวสุ่มที่ใช้ Seed 42 เพื่อให้สุ่มได้ผลเดิมทุกครั้ง
        rng = random.Random(42)
        # สร้าง Dictionary ว่างสำหรับเก็บรายชื่อภาพในแต่ละ Split
        selected = {}
        # วนเลือกชื่อ Split ไฟล์รายการ และจำนวนภาพของ Validation กับ Test
        for split, filename, count in [("validation", "trainval.txt", 20), ("test", "test.txt", 50)]:
            # เปิดไฟล์รายการภายใน Archive โดยไม่ต้องแตก Archive ทั้งหมด
            with archive.extractfile("annotations/" + filename) as source:
                # ถอดรหัสแต่ละบรรทัดและเก็บเฉพาะชื่อภาพจากบรรทัดที่ไม่ว่าง
                names = [line.decode().split()[0] for line in source if line.strip()]
            # สุ่มชื่อภาพตามจำนวนที่กำหนด เรียงชื่อ แล้วเก็บลง Split ปัจจุบัน
            selected[split] = sorted(rng.sample(names, count))
        # รวมรายชื่อ Validation และ Test เพื่อเตรียมดึงไฟล์ทั้งหมด
        names = selected["validation"] + selected["test"]
        # วนทำงานกับภาพแต่ละชื่อที่ถูกเลือก
        for name in names:
            # สร้างพาธปลายทางของไฟล์ Trimap ตามชื่อภาพ
            target = DATA / "masks" / (name + ".png")
            # สร้างโฟลเดอร์ masks หากยังไม่มีอยู่
            target.parent.mkdir(exist_ok=True)
            # เปิดไฟล์ Trimap ของภาพปัจจุบันจากภายใน Archive
            with archive.extractfile(f"annotations/trimaps/{name}.png") as source:
                # อ่านข้อมูลไบต์ทั้งหมดแล้วบันทึกเป็นไฟล์ Mask ปลายทาง
                target.write_bytes(source.read())
    # อธิบายว่าการอ่านแบบ Stream ป้องกันการแตกไฟล์ภาพที่ไม่ได้เลือก
    # อ่าน archive แบบ stream โดยไม่แตกไฟล์ภาพที่ไม่ใช้
    # ดาวน์โหลดไฟล์ Archive ที่เก็บภาพต้นฉบับทั้งหมด
    images = download("images.tar.gz")
    # สร้าง Set ของพาธภาพภายใน Archive ที่ต้องการดึงออกมา
    wanted = {f"images/{name}.jpg" for name in names}
    # เปิด Archive รูปภาพแบบ Stream และคลายข้อมูลตามลำดับ
    with tarfile.open(images, "r|gz") as archive:
        # วนตรวจสมาชิกแต่ละรายการภายใน Archive
        for member in archive:
            # ทำงานต่อเมื่อสมาชิกปัจจุบันเป็นรูปภาพที่ต้องการ
            if member.name in wanted:
                # สร้างพาธปลายทางโดยอิงจากชื่อสมาชิกใน Archive
                target = DATA / member.name
                # สร้างโฟลเดอร์แม่ของรูปภาพหากยังไม่มีอยู่
                target.parent.mkdir(exist_ok=True)
                # เปิดข้อมูลรูปภาพปัจจุบันจาก Archive
                with archive.extractfile(member) as source:
                    # อ่านข้อมูลไบต์แล้วบันทึกเป็นไฟล์รูปภาพปลายทาง
                    target.write_bytes(source.read())
                # ลบชื่อภาพที่ดาวน์โหลดสำเร็จออกจาก Set ที่ยังรออยู่
                wanted.remove(member.name)
                # ตรวจสอบว่าดึงภาพที่ต้องการครบแล้วหรือไม่
                if not wanted:
                    # หยุดอ่าน Archive ทันทีเมื่อได้รูปภาพครบทั้งหมด
                    break
    # ตรวจสอบหลังอ่าน Archive ว่ายังมีรูปภาพใดหาไม่พบหรือไม่
    if wanted:
        # แจ้งข้อผิดพลาดพร้อมรายชื่อรูปภาพที่ขาดหาย
        raise RuntimeError(f"Missing images: {sorted(wanted)}")
    # บันทึกรายชื่อ Validation และ Test เป็น JSON แบบเยื้องให้อ่านง่าย
    (DATA / "split.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    # แสดงข้อความยืนยันว่าจัดเตรียมภาพและ Trimap ครบแล้ว
    print("Ready: 20 validation images + 50 test images, each with a trimap.")


# =============================================================================
# ส่วนที่ 5: จุดเริ่มต้นการทำงานเมื่อรันไฟล์โดยตรง
# =============================================================================
# ตรวจสอบว่าไฟล์นี้ถูกรันโดยตรงแทนการถูก Import เป็นโมดูล
if __name__ == "__main__":
    # เรียกฟังก์ชันหลักเพื่อเริ่มดาวน์โหลดและเตรียมข้อมูล
    main()
