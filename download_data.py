# =============================================================================
# ส่วนที่ 1: คำอธิบายโมดูลและไลบรารีที่ใช้
# =============================================================================
# ระบุว่าโมดูลนี้ใช้สร้างชุดข้อมูลภาพแอปเปิลเดี่ยวบนพื้นผิวธรรมชาติพร้อมเงาทอดตกกระทบ (Realistic Difficulty)
"""สร้างชุดข้อมูลภาพแอปเปิลเดี่ยวบนพื้นผิวธรรมชาติ (ไม้/หินอ่อน) พร้อมเงาตกกระทบสมจริง."""
from pathlib import Path
import random
import urllib.request
import json
import cv2
import numpy as np
from PIL import Image
from json_comments import write_commented_json

# =============================================================================
# ส่วนที่ 2: การกำหนดพาธและแหล่งดาวน์โหลดชุดข้อมูล
# =============================================================================
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW_CACHE = DATA / ".raw_apples"
# แหล่งดาวน์โหลดชุดข้อมูลภาพแอปเปิลสีแดง 5 สายพันธุ์จาก Fruits-360
RAW_BASE_TEMPLATE = "https://raw.githubusercontent.com/Horea94/Fruit-Images-Dataset/master/Training/{variety}/{filename}"

# รายการภาพแอปเปิลสีแดง 5 สายพันธุ์ (สายพันธุ์ละ 14 ภาพ = รวม 70 ภาพ)
# 1. Apple Red 1 (แดงสด)
# 2. Apple Red 2 (แดงทับทิม)
# 3. Apple Red 3 (แดงเข้ม)
# 4. Apple Braeburn (แดงลายเสี้ยนธรรมชาติ)
# 5. Apple Crimson Snow (แดงคริมสัน)
VARIETIES_CONFIG = [
    {
        "variety": "Apple%20Red%201",
        "name_prefix": "red1",
        "files": ['0_100.jpg', '1_100.jpg', '2_100.jpg', '10_100.jpg', '11_100.jpg', '12_100.jpg', '13_100.jpg',
                  '14_100.jpg', '15_100.jpg', '16_100.jpg', '17_100.jpg', '18_100.jpg', '19_100.jpg', '20_100.jpg']
    },
    {
        "variety": "Apple%20Red%202",
        "name_prefix": "red2",
        "files": ['0_100.jpg', '1_100.jpg', '2_100.jpg', '10_100.jpg', '11_100.jpg', '12_100.jpg', '13_100.jpg',
                  '14_100.jpg', '15_100.jpg', '16_100.jpg', '17_100.jpg', '18_100.jpg', '19_100.jpg', '20_100.jpg']
    },
    {
        "variety": "Apple%20Red%203",
        "name_prefix": "red3",
        "files": ['0_100.jpg', '1_100.jpg', '2_100.jpg', '3_100.jpg', '10_100.jpg', '11_100.jpg', '12_100.jpg',
                  '13_100.jpg', '14_100.jpg', '15_100.jpg', '16_100.jpg', '17_100.jpg', '18_100.jpg', '19_100.jpg']
    },
    {
        "variety": "Apple%20Braeburn",
        "name_prefix": "braeburn",
        "files": ['0_100.jpg', '1_100.jpg', '2_100.jpg', '10_100.jpg', '11_100.jpg', '12_100.jpg', '13_100.jpg',
                  '14_100.jpg', '15_100.jpg', '16_100.jpg', '17_100.jpg', '18_100.jpg', '19_100.jpg', '20_100.jpg']
    },
    {
        "variety": "Apple%20Crimson%20Snow",
        "name_prefix": "crimson",
        "files": ['0_100.jpg', '1_100.jpg', '2_100.jpg', '3_100.jpg', '4_100.jpg', '5_100.jpg', '6_100.jpg',
                  '7_100.jpg', '8_100.jpg', '9_100.jpg', '10_100.jpg', '11_100.jpg', '12_100.jpg', '13_100.jpg']
    }
]


# =============================================================================
# ส่วนที่ 3: การสร้างพื้นผิวธรรมชาติสมจริง (Wood & Stone Surfaces)
# =============================================================================
def generate_surface(h=220, w=220, surface_type="oak", seed=0):
    """สร้างภาพพื้นผิวธรรมชาติ (ไม้โอ๊ก, ไม้สน, หินอ่อน, ไม้วอลนัต) พร้อมเกรเดียนต์แสงสมจริง."""
    rng = np.random.RandomState(seed)
    y_coords, x_coords = np.mgrid[0:h, 0:w]

    # เกรเดียนต์แสงตกกระทบตามธรรมชาติ (มุมแสงเอียงเล็กน้อย)
    light_angle = rng.uniform(0.7, 1.1)
    light = 1.0 - 0.20 * ((x_coords * 0.7 + y_coords * 0.9) / (h + w)) * light_angle

    if surface_type == "oak":
        # พื้นไม้โอ๊กโทนสีน้ำตาลทอง ลายเสี้ยนไม้แนวนอน
        grain_freq = 0.10 + rng.uniform(-0.015, 0.015)
        wobble = 4.0 * np.sin(y_coords * 0.04)
        grain = 0.5 + 0.5 * np.sin((x_coords + wobble) * grain_freq)
        grain = np.power(grain, 1.8) * 0.16
        r = (202 * light - 28 * grain + rng.normal(0, 1.6, (h, w))).clip(0, 255)
        g = (166 * light - 24 * grain + rng.normal(0, 1.6, (h, w))).clip(0, 255)
        b = (128 * light - 18 * grain + rng.normal(0, 1.6, (h, w))).clip(0, 255)
    elif surface_type == "pine":
        # พื้นไม้สนสีเหลืองอำพัน สว่างและอบอุ่น
        grain_freq = 0.08
        wobble = 3.0 * np.cos(x_coords * 0.03)
        grain = 0.5 + 0.5 * np.sin((y_coords + wobble) * grain_freq)
        grain = np.power(grain, 1.5) * 0.13
        r = (222 * light - 24 * grain + rng.normal(0, 1.5, (h, w))).clip(0, 255)
        g = (190 * light - 21 * grain + rng.normal(0, 1.5, (h, w))).clip(0, 255)
        b = (145 * light - 17 * grain + rng.normal(0, 1.5, (h, w))).clip(0, 255)
    elif surface_type == "marble":
        # พื้นหินอ่อน/เคาน์เตอร์หิน ลายริ้วจางๆ
        freq1 = 0.03
        freq2 = 0.06
        veins = np.sin(x_coords * freq1 + np.sin(y_coords * freq2) * 2.0)
        veins = np.power(np.abs(veins), 3.5) * 0.14
        r = (205 * light - 22 * veins + rng.normal(0, 1.8, (h, w))).clip(0, 255)
        g = (200 * light - 21 * veins + rng.normal(0, 1.8, (h, w))).clip(0, 255)
        b = (194 * light - 19 * veins + rng.normal(0, 1.8, (h, w))).clip(0, 255)
    else:  # walnut
        # พื้นไม้วอลนัตสีน้ำตาลเข้มหรูหรา
        grain_freq = 0.11
        grain = 0.5 + 0.5 * np.sin(x_coords * grain_freq + rng.normal(0, 0.4, (h, w)))
        grain = np.power(grain, 1.7) * 0.18
        r = (162 * light - 32 * grain + rng.normal(0, 1.6, (h, w))).clip(0, 255)
        g = (126 * light - 26 * grain + rng.normal(0, 1.6, (h, w))).clip(0, 255)
        b = (96 * light - 20 * grain + rng.normal(0, 1.6, (h, w))).clip(0, 255)

    return np.stack([r, g, b], axis=-1).astype(np.uint8)


# =============================================================================
# ส่วนที่ 4: การวางผลแอปเปิลพร้อมสร้างเงาทอดตกกระทบและ Trimap Mask
# =============================================================================
def composite_apple_on_surface(apple_rgb, bg, seed=0):
    """ประกอบผลแอปเปิลลงบนพื้นผิว พร้อมเงาทอดตกกระทบสมจริง และสร้าง Ground truth Trimap."""
    rng = np.random.RandomState(seed)
    h_bg, w_bg = bg.shape[:2]

    # 1. แยกเนื้อแอปเปิลออกจากภาพดิบพื้นหลังขาว (Erode 2px เพื่อกำจัดขอบขาวเดิมให้เนียนสนิท)
    dist_white = np.linalg.norm(apple_rgb.astype(np.float32) - 255.0, axis=2)
    apple_mask = (dist_white > 35.0).astype(np.uint8)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    apple_mask = cv2.erode(apple_mask, kernel_small, iterations=2)
    apple_mask_feather = cv2.GaussianBlur(apple_mask.astype(np.float32), (3, 3), 0.6)

    target_size = 140
    apple_scaled = cv2.resize(apple_rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
    mask_scaled = cv2.resize(apple_mask_feather, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # วางตำแหน่งกึ่งกลางพร้อมสุ่มเลื่อนเล็กน้อยอย่างเป็นธรรมชาติ
    off_y = (h_bg - target_size) // 2 + rng.randint(-10, 11)
    off_x = (w_bg - target_size) // 2 + rng.randint(-10, 11)

    canvas_mask = np.zeros((h_bg, w_bg), dtype=np.float32)
    canvas_mask[off_y:off_y+target_size, off_x:off_x+target_size] = mask_scaled

    # 2. สร้างเงาทอดตกกระทบ (Directional Cast Shadow) ใต้ผลแอปเปิล
    shadow_mask = np.zeros((h_bg, w_bg), dtype=np.float32)
    shadow_dx = rng.choice([10, 14, -10, -14])
    shadow_dy = rng.randint(8, 14)
    center = (off_x + target_size // 2 + shadow_dx, off_y + target_size - 4 + shadow_dy)
    axes = (target_size // 2 + 12, target_size // 4 + 6)
    angle = 12 if shadow_dx > 0 else -12
    cv2.ellipse(shadow_mask, center, axes, angle, 0, 360, 1.0, -1)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (35, 35), 9.0)

    # เงาทำให้พื้นผิวมืดลง 38% - 46% (สร้างความท้าทายจริงให้แก่ระบบตัดแบ่งสี)
    shadow_opacity = rng.uniform(0.38, 0.46)
    shadow_mask = shadow_mask * (1.0 - np.clip(canvas_mask * 1.5, 0.0, 1.0)) * shadow_opacity

    # นำเงาไปผสานลงบนพื้นผิวฉากหลัง
    result_bg = bg.astype(np.float32) * (1.0 - shadow_mask[:, :, None])

    # 3. วางผลแอปเปิลทับลงบนพื้นผิว
    apple_full = np.zeros((h_bg, w_bg, 3), dtype=np.float32)
    apple_full[off_y:off_y+target_size, off_x:off_x+target_size] = apple_scaled.astype(np.float32)

    alpha_3d = canvas_mask[:, :, None]
    comp = (apple_full * alpha_3d + result_bg * (1.0 - alpha_3d)).clip(0, 255).astype(np.uint8)

    # 4. สร้าง Ground truth Trimap:
    # 1 = Apple (เนื้อผลไม้จริง)
    # 2 = Background (พื้นผิวโต๊ะ + เงาทอดตกกระทบ)
    # 3 = Boundary (ขอบรอยต่อ 1-2 พิกเซลเพื่อไม่คิดคะแนนความคลุมเครือ)
    binary_apple = (canvas_mask > 0.5).astype(np.uint8)
    dilated = cv2.dilate(binary_apple, kernel_small, iterations=1)
    eroded = cv2.erode(binary_apple, kernel_small, iterations=1)

    trimap = np.full((h_bg, w_bg), 2, dtype=np.uint8)
    trimap[binary_apple == 1] = 1
    trimap[dilated != eroded] = 3

    return comp, trimap


# =============================================================================
# ส่วนที่ 5: การประมวลผลและจัดเตรียมชุดข้อมูลทั้ง 70 ภาพ
# =============================================================================
# =============================================================================
# ส่วนที่ 5: การประมวลผลและจัดเตรียมชุดข้อมูลแอปเปิลสีแดง 5 สไตล์ทั้ง 70 ภาพ
# =============================================================================
def main():
    print("Preparing Red Apples Dataset (One Color, 5 Styles) on Textured Surfaces...", flush=True)

    img_dir = DATA / "images"
    mask_dir = DATA / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    RAW_CACHE.mkdir(parents=True, exist_ok=True)

    # จัดเตรียม 20 ภาพสำหรับ Validation (สายพันธุ์ละ 4 ภาพ)
    # และ 50 ภาพสำหรับ Test (สายพันธุ์ละ 10 ภาพ)
    val_names = []
    test_names = []
    task_items = []  # (url, cache_name, new_name, variety_name)

    val_counter = 1
    test_counter = 21

    for v_conf in VARIETIES_CONFIG:
        variety = v_conf["variety"]
        prefix = v_conf["name_prefix"]
        files = v_conf["files"]

        # 4 ภาพแรกสำหรับ Validation
        for f in files[:4]:
            new_name = f"apple_{val_counter:02d}"
            val_names.append(new_name)
            url = RAW_BASE_TEMPLATE.format(variety=variety, filename=f)
            task_items.append((url, f"{prefix}_{f}", new_name, variety))
            val_counter += 1

        # 10 ภาพถัดไปสำหรับ Test
        for f in files[4:14]:
            new_name = f"apple_{test_counter:02d}"
            test_names.append(new_name)
            url = RAW_BASE_TEMPLATE.format(variety=variety, filename=f)
            task_items.append((url, f"{prefix}_{f}", new_name, variety))
            test_counter += 1

    surface_types = ["oak", "pine", "marble", "walnut"]
    headers = {"User-Agent": "Mozilla/5.0"}

    for idx, (url, cache_filename, new_name, variety) in enumerate(task_items):
        cached_raw = RAW_CACHE / cache_filename
        if not cached_raw.exists():
            print(f"[{idx + 1}/70] Downloading {variety} ({cache_filename}) ...", flush=True)
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as resp:
                data = resp.read()
            nparr = np.frombuffer(data, np.uint8)
            bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            raw_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            Image.fromarray(raw_rgb).save(cached_raw, format="JPEG", quality=95)
        else:
            with Image.open(cached_raw) as img:
                raw_rgb = np.array(img.convert("RGB"))

        # สลับชนิดพื้นผิว (ไม้โอ๊ก, ไม้สน, หินอ่อน, ไม้วอลนัต) เพื่อความหลากหลายสมจริง
        surface_type = surface_types[idx % len(surface_types)]
        bg = generate_surface(h=220, w=220, surface_type=surface_type, seed=idx + 100)

        # ผสานแอปเปิล เงาตกกระทบ และ Trimap
        comp, trimap = composite_apple_on_surface(raw_rgb, bg, seed=idx + 200)

        # บันทึกภาพผลลัพธ์และ Mask
        img_path = img_dir / f"{new_name}.jpg"
        mask_path = mask_dir / f"{new_name}.png"

        Image.fromarray(comp).save(img_path, format="JPEG", quality=95)
        Image.fromarray(trimap).save(mask_path, format="PNG")

        if (idx + 1) % 10 == 0 or idx == 69:
            print(f"Generated {idx + 1}/70 images ({new_name}, {variety}, Surface: {surface_type})...", flush=True)

    # บันทึก split.json
    split_data = {
        "validation": val_names,
        "test": test_names
    }
    write_commented_json(DATA / "split.json", split_data)
    print(f"Successfully generated 70 Red Apple images (20 val, 50 test) in {DATA}!", flush=True)


if __name__ == "__main__":
    main()
