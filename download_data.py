"""ดาวน์โหลดภาพและ Ground truth จาก Oxford-IIIT Pet (ทำครั้งแรกครั้งเดียว)."""
from pathlib import Path
import random
import tarfile
import urllib.request
import json

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SOURCE = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/"


def download(name):
    """เก็บ archive ไว้ใช้ซ้ำ; เปลี่ยนชื่อเมื่อดาวน์โหลดสำเร็จเท่านั้น."""
    target = DATA / name
    if not target.exists():
        print(f"Downloading {name} ...", flush=True)
        partial = target.with_suffix(".part")
        urllib.request.urlretrieve(SOURCE + name, partial)
        partial.replace(target)
    return target


def main():
    DATA.mkdir(exist_ok=True)
    annotations = download("annotations.tar.gz")
    # ใช้รายการ trainval เลือก 20 ภาพปรับ threshold และ test เลือก 50 ภาพทดสอบ
    with tarfile.open(annotations) as archive:
        rng = random.Random(42)
        selected = {}
        for split, filename, count in [("validation", "trainval.txt", 20), ("test", "test.txt", 50)]:
            with archive.extractfile("annotations/" + filename) as source:
                names = [line.decode().split()[0] for line in source if line.strip()]
            selected[split] = sorted(rng.sample(names, count))
        names = selected["validation"] + selected["test"]
        for name in names:
            target = DATA / "masks" / (name + ".png")
            target.parent.mkdir(exist_ok=True)
            with archive.extractfile(f"annotations/trimaps/{name}.png") as source:
                target.write_bytes(source.read())
    # อ่าน archive แบบ stream โดยไม่แตกไฟล์ภาพที่ไม่ใช้
    images = download("images.tar.gz")
    wanted = {f"images/{name}.jpg" for name in names}
    with tarfile.open(images, "r|gz") as archive:
        for member in archive:
            if member.name in wanted:
                target = DATA / member.name
                target.parent.mkdir(exist_ok=True)
                with archive.extractfile(member) as source:
                    target.write_bytes(source.read())
                wanted.remove(member.name)
                if not wanted:
                    break
    if wanted:
        raise RuntimeError(f"Missing images: {sorted(wanted)}")
    (DATA / "split.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print("Ready: 20 validation images + 50 test images, each with a trimap.")


if __name__ == "__main__":
    main()
