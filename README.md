# Segmentation Workshop: แยกแมวและสุนัขออกจากพื้นหลัง

โปรเจกต์ Python แบบพื้นฐานสำหรับงานกลุ่ม Image Segmentation ใช้ความต่างของสีจากขอบภาพ โดยไม่ใช้ Deep Learning พร้อม Morphology, Confusion Matrix และ ROC Curve

## เริ่มใช้งาน

ต้องมี Python 3.13 และอินเทอร์เน็ตสำหรับดาวน์โหลดครั้งแรก เปิด terminal ในโฟลเดอร์นี้แล้วรันตามลำดับ (Windows PowerShell):

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe download_data.py
.venv/Scripts/python.exe run.py
.venv/Scripts/python.exe -m unittest -v
```

บน macOS/Linux ใช้ `.venv/bin/python` แทน `.venv/Scripts/python.exe`

การดาวน์โหลดต้นฉบับใช้ประมาณ 800 MB แม้เลือกใช้เพียง 70 ภาพ เนื่องจากต้นทางจัดเก็บเป็น archive รวม โปรแกรมเก็บเฉพาะภาพและ mask ที่เลือก แต่เก็บ archive ไว้ด้วยเพื่อไม่ต้องดาวน์โหลดซ้ำ ข้อมูลอยู่ใน `data/` และไม่ถูกส่งขึ้น Git

## งานนี้ทำอะไร

| ข้อกำหนด | สิ่งที่โปรเจกต์ทำ |
|---|---|
| เลือกโจทย์ Segmentation | แยกตัวแมวหรือสุนัขออกจากพื้นหลัง |
| อย่างน้อย 50 ภาพและ Ground truth | 70 คู่ภาพ/trimap: validation 20 ภาพและ test 50 ภาพ |
| ออกแบบ Algorithm | สี Lab → สี median ที่ขอบ → ระยะห่างสี → threshold |
| ทดลอง Morphology | Opening แล้ว Closing ด้วย kernel วงรี 3×3 |
| Confusion Matrix และ ROC | ประเมินพิกเซลจริง เทียบก่อน/หลัง Morphology และสร้าง ROC จากคะแนนต่อเนื่อง |
| นำเสนอ | ใช้คำอธิบายและแนวทางนำเสนอใน [EXPLANATION_TH.md](EXPLANATION_TH.md) พร้อมผลใน outputs |

## อ่านโค้ดตามลำดับนี้

1. [segmentation.py](segmentation.py): สูตรคะแนน, Morphology และสูตรประเมินผล
2. [download_data.py](download_data.py): ดาวน์โหลดและจัดชุดข้อมูล
3. [run.py](run.py): เรียกทุกส่วนให้ทำงานต่อกัน และสร้างกราฟ
4. [EXPLANATION_TH.md](EXPLANATION_TH.md): อธิบายแต่ละฟังก์ชันและวิธีอ่านผล

## ผลลัพธ์

หลังรัน `run.py` เปิดโฟลเดอร์ `outputs/`:

- `RESULTS.md`: ตารางสรุปผลจริง
- `confusion_matrix.png`: จำนวน TN, FP, FN, TP ก่อนและหลัง Morphology
- `roc_curve.png`: ROC และ AUC จากคะแนนความต่างสี พร้อมจุดทำงานก่อน/หลัง
- `examples/`: ภาพเปรียบเทียบครบทั้ง 50 ภาพ ได้แก่ภาพจริง, เฉลย, mask ก่อน/หลัง และภาพตัดพื้นหลัง
- `per_image.csv`: ผลแยกเป็นรายภาพ
- `summary.json`: ค่าประเมินรวม
- `threshold_search.json`: ผลเลือก threshold บน validation
- `split.json`: รายชื่อภาพที่ใช้ เพื่อทำซ้ำและตรวจสอบได้

## ข้อมูลและข้อจำกัด

ใช้ [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) ของ O. M. Parkhi, A. Vedaldi, A. Zisserman และ C. V. Jawahar, *Cats and Dogs*, CVPR 2012. ต้นทางระบุสัญญาอนุญาต CC BY-SA 4.0; ลิขสิทธิ์ภาพยังเป็นของเจ้าของเดิม ภาพตัวอย่างที่สร้างจากชุดข้อมูลใน `outputs/examples/` ต้องให้เครดิตต้นทางและอยู่ภายใต้เงื่อนไขของชุดข้อมูลด้วย

Trimap: 1 = สัตว์, 2 = พื้นหลัง, 3 = ขอบที่ไม่แน่นอน โค้ดไม่นำ label 3 มาประเมิน และแสดงบริเวณนั้นเป็นสีเทาในภาพเฉลย

นี่เป็น baseline เพื่อเรียนรู้ ไม่ใช่ระบบแยกสัตว์ที่แม่นยำสูง วิธีนี้สมมติว่าขอบภาพเป็นพื้นหลัง จึงอาจผิดเมื่อสัตว์อยู่ติดขอบหรือสีใกล้พื้นหลัง Morphology อาจทำให้ผลดีขึ้นหรือแย่ลง ต้องดูผลจริง ผลที่รายงานเป็นการทดลองย่อภาพด้านยาวไม่เกิน 256 พิกเซล ไม่ใช่ benchmark ที่ความละเอียดต้นฉบับ
