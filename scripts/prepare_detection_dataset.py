"""准备车辆检测数据集：从 COCO val2017 选图、转 YOLO 格式、划分 train/val，并打包成 zip。

用法：
    python scripts/prepare_detection_dataset.py
"""
from __future__ import annotations

import json
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(r"D:\stage2\datasets\vehicle_det")
ZIP = Path(r"D:\stage2\data\coco\annotations_trainval2017.zip")
TEST_DIR = Path(r"D:\stage2\data\test_images")
WEIGHTS = Path(r"D:\stage2\weights\yolo26n.pt")
OUT_ZIP = Path(r"D:\stage2\datasets\vehicle_det_kaggle.zip")

N = 300
SEED = 42
CAT = {"car": 0, "truck": 1, "bus": 2, "motorcycle": 3}


def main() -> None:
    with zipfile.ZipFile(ZIP) as z, z.open("annotations/instances_val2017.json") as f:
        data = json.load(f)
    cats = {c["id"]: c["name"] for c in data["categories"]}
    files = {im["id"]: im["file_name"] for im in data["images"]}
    sizes = {im["id"]: (im["width"], im["height"]) for im in data["images"]}
    exclude = {p.name for p in TEST_DIR.glob("*.jpg")}

    boxes: dict[int, list] = {}
    for a in data["annotations"]:
        if a.get("iscrowd"):
            continue
        name = cats[a["category_id"]]
        if name in CAT:
            boxes.setdefault(a["image_id"], []).append((CAT[name], a["bbox"]))

    cands = [iid for iid, bs in boxes.items() if files[iid] not in exclude and bs]
    rng = random.Random(SEED)
    rng.shuffle(cands)
    picks = cands[:N]
    rng.shuffle(picks)
    n_val = int(len(picks) * 0.2)
    val_ids = set(picks[:n_val])
    print(f"选中 {len(picks)} 张（val {n_val} / train {len(picks) - n_val}）")

    def download(iid):
        fn = files[iid]
        split = "val" if iid in val_ids else "train"
        img_dir = ROOT / "images" / split
        lab_dir = ROOT / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        dst = img_dir / fn
        if not dst.exists():
            r = requests.get(f"http://images.cocodataset.org/val2017/{fn}", timeout=60)
            r.raise_for_status()
            dst.write_bytes(r.content)
        W, H = sizes[iid]
        lines = []
        for cls, (x, y, w, h) in boxes[iid]:
            cx, cy = (x + w / 2) / W, (y + h / 2) / H
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")
        (lab_dir / f"{dst.stem}.txt").write_text("\n".join(lines), encoding="utf-8")

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(download, picks))

    (ROOT / "vehicle.yaml").write_text(
        "path: vehicle_det\ntrain: images/train\nval: images/val\n"
        "names:\n  0: car\n  1: truck\n  2: bus\n  3: motorcycle\n",
        encoding="utf-8")

    OUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for p in ROOT.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(ROOT.parent))
        z.write(WEIGHTS, "yolo26n.pt")

    print(f"数据集: {ROOT}")
    print(f"打包: {OUT_ZIP}  ({OUT_ZIP.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
