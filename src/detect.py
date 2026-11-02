"""YOLO 车辆检测：输入图片或文件夹，输出车辆计数、标注图和 JSON。

用法：
    python -m src.detect --source assets
    python -m src.detect --source data/test_images --conf 0.3
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle", "bicycle"}
WEIGHTS = Path(r"D:\stage2\weights\yolo11n.pt")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def detect_image(model: YOLO, img_path: Path, conf: float):
    results = model(str(img_path), conf=conf, verbose=False)
    boxes = results[0].boxes
    dets = []
    counts = Counter()
    for i in range(len(boxes)):
        name = model.names[int(boxes.cls[i])]
        if name not in VEHICLE_CLASSES:
            continue
        dets.append({
            "class": name,
            "confidence": round(float(boxes.conf[i]), 3),
            "box": [round(float(v), 1) for v in boxes.xyxy[i]],
        })
        counts[name] += 1
    return dets, dict(counts), results[0].orig_img


def draw(img, dets):
    vis = img.copy()
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{d["class"]} {d["confidence"]:.2f}'
        cv2.putText(vis, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return vis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="图片或文件夹")
    ap.add_argument("--out", default=r"D:\stage2\results\detect")
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()

    model = YOLO(str(WEIGHTS))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = Path(args.source)
    files = [src] if src.is_file() else sorted(
        p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)

    all_results = {}
    for f in files:
        dets, counts, img = detect_image(model, f, args.conf)
        cv2.imwrite(str(out_dir / f"{f.stem}_detect.jpg"), draw(img, dets))
        total = sum(counts.values())
        all_results[f.name] = {"counts": counts, "total": total, "detections": dets}
        print(f"{f.name}: {counts}  total={total}")

    (out_dir / "detect_results.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
