"""从 COCO val2017 里挑选车辆数量适中的道路场景图，下载图片并生成 ground_truth.csv。

用法：
    python scripts/prepare_coco.py

产物：
    data/test_images/*.jpg
    data/ground_truth.csv   （filename, cars, buses, trucks, motorcycles, total）
"""
from __future__ import annotations

import json
import random
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(r"D:\stage2\data\coco")
ZIP = ROOT / "annotations_trainval2017.zip"
OUT_IMAGES = Path(r"D:\stage2\data\test_images")
GT = Path(r"D:\stage2\data\ground_truth.csv")

N = 40
VEH = ("car", "bus", "truck", "motorcycle")
random.seed(42)


def main() -> None:
    with zipfile.ZipFile(ZIP) as z:
        with z.open("annotations/instances_val2017.json") as f:
            data = json.load(f)
    cats = {c["id"]: c["name"] for c in data["categories"]}
    imgs = {im["id"]: im for im in data["images"]}

    per_img: dict[int, Counter] = {}
    for a in data["annotations"]:
        if a.get("iscrowd"):
            continue
        name = cats[a["category_id"]]
        if name in VEH:
            per_img.setdefault(a["image_id"], Counter())[name] += 1

    # 挑车辆数量 2~10、且目标不太小的图（太小 YOLO 容易漏检，对比不公平）
    size_info = {im["id"]: (im["width"], im["height"]) for im in data["images"]}
    area_by_img: dict[int, list[float]] = {}
    for a in data["annotations"]:
        if a.get("iscrowd"):
            continue
        if cats[a["category_id"]] not in VEH:
            continue
        W, H = size_info[a["image_id"]]
        area_by_img.setdefault(a["image_id"], []).append(a["bbox"][2] * a["bbox"][3] / (W * H))

    cands = []
    for iid, c in per_img.items():
        n = sum(c.values())
        if not (4 <= n <= 10):
            continue
        areas = sorted(area_by_img.get(iid, [0]))
        med = areas[len(areas) // 2]
        if med >= 0.006:
            cands.append((iid, c))
    random.shuffle(cands)
    picks = cands[:N]
    print(f"候选 {len(cands)} 张，选中 {len(picks)} 张")

    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    rows = []
    for iid, c in picks:
        fname = imgs[iid]["file_name"]
        dst = OUT_IMAGES / fname
        if not dst.exists():
            url = f"http://images.cocodataset.org/val2017/{fname}"
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            dst.write_bytes(r.content)
        rows.append(dict(
            filename=fname,
            cars=c.get("car", 0),
            buses=c.get("bus", 0),
            trucks=c.get("truck", 0),
            motorcycles=c.get("motorcycle", 0),
            total=sum(c.values()),
        ))
        print(f"{fname}: {dict(c)}")

    gt = pd.DataFrame(rows)
    gt.to_csv(GT, index=False, encoding="utf-8")
    print(f"标注表 -> {GT}, 共 {len(gt)} 行")
    print(f"车辆总数分布: min={gt.total.min()} max={gt.total.max()} mean={gt.total.mean():.1f}")


if __name__ == "__main__":
    main()
