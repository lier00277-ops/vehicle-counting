"""并排对比图：原图、YOLO 检测结果、大模型结果。"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATA = Path(r"D:\stage2\data\test_images")
TRAD = Path(r"D:\stage2\results\traditional")
GT = Path(r"D:\stage2\data\ground_truth.csv")
YOLO_JSON = Path(r"D:\stage2\results\traditional\results.json")
VLM_JSON = Path(r"D:\stage2\results\vlm_results.json")
OUT = Path(r"D:\stage2\figures\comparison_samples.png")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    gt = pd.read_csv(GT)
    yolo = json.load(open(YOLO_JSON, encoding="utf-8"))
    vlm = json.load(open(VLM_JSON, encoding="utf-8"))

    gt_sorted = gt.sort_values("total").reset_index(drop=True)
    idx = [int(i * (len(gt_sorted) - 1) / 5) for i in range(6)]
    picks = gt_sorted.iloc[idx]["filename"].tolist()

    fig, axes = plt.subplots(len(picks), 3, figsize=(15, 3.6 * len(picks)))
    for r, name in enumerate(picks):
        img = cv2.cvtColor(cv2.imread(str(DATA / name)), cv2.COLOR_BGR2RGB)
        ypath = TRAD / f"{Path(name).stem}_detect.jpg"
        yimg = cv2.cvtColor(cv2.imread(str(ypath)), cv2.COLOR_BGR2RGB) if ypath.exists() else img

        g = int(gt[gt.filename == name]["total"].iloc[0])
        y = sum(yolo.get(name, {}).get("counts", {}).values())
        rec = vlm.get(name, {}).get("parsed") or {}
        v = rec.get("total", "?")

        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"原图（真值 {g} 辆）", fontsize=9)
        axes[r, 0].axis("off")

        axes[r, 1].imshow(yimg)
        axes[r, 1].set_title(f"YOLO：{y} 辆", fontsize=9)
        axes[r, 1].axis("off")

        axes[r, 2].axis("off")
        desc = rec.get("description", "")
        detail = f"cars={rec.get('cars', '?')}  buses={rec.get('buses', '?')}  " \
                 f"trucks={rec.get('trucks', '?')}  motorcycles={rec.get('motorcycles', '?')}"
        axes[r, 2].text(0.02, 0.5, f"VLM：{v} 辆\n{detail}\n\n{desc}",
                        fontsize=8, va="center")

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
