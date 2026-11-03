"""传统方案：YOLO 检测车辆，可选 SAM 分割，输出计数、结果图和 JSON。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.segment import load_predictor

WEIGHTS = Path(r"D:\stage2\weights\yolo26n.pt")
VEH = {"car", "bus", "truck", "motorcycle"}
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=r"D:\stage2\data\test_images")
    ap.add_argument("--out", default=r"D:\stage2\results\traditional")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--sam", action="store_true", help="用 SAM 分割每辆车")
    args = ap.parse_args()

    model = YOLO(str(WEIGHTS))
    predictor = load_predictor() if args.sam else None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(args.source)
    files = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)

    results = {}
    for f in files:
        r = model(str(f), conf=args.conf, verbose=False)[0]
        img = r.orig_img
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        dets = []
        for i in range(len(r.boxes)):
            name = model.names[int(r.boxes.cls[i])]
            if name not in VEH:
                continue
            dets.append({
                "class": name,
                "confidence": round(float(r.boxes.conf[i]), 3),
                "box": [round(float(v), 1) for v in r.boxes.xyxy[i]],
            })

        vis = img.copy()
        if predictor is not None:
            predictor.set_image(rgb)
            for d in dets:
                masks, _, _ = predictor.predict(box=np.array(d["box"]), multimask_output=False)
                d["mask_area"] = int(masks[0].sum())
                vis[masks[0]] = (vis[masks[0]] * 0.55 + np.array([255, 80, 0]) * 0.45).astype(np.uint8)

        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d["box"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(vis, f'{d["class"]} {d["confidence"]:.2f}', (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        counts: dict[str, int] = {}
        for d in dets:
            counts[d["class"]] = counts.get(d["class"], 0) + 1
        total = sum(counts.values())
        results[f.name] = {"counts": counts, "total": total, "detections": dets}
        cv2.imwrite(str(out_dir / f"{f.stem}_detect.jpg"), vis)
        print(f"{f.name}: {counts} total={total}")

    (out_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved ->", out_dir)


if __name__ == "__main__":
    main()
