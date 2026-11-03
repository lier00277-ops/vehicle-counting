"""生成对比图表：MAE 柱状图、误差分布、检测示例。"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FIG = Path(r"D:\stage2\figures")
TRAD = Path(r"D:\stage2\results\traditional")
GT = Path(r"D:\stage2\data\ground_truth.csv")
CMP = Path(r"D:\stage2\results\comparison.csv")
YOLO_JSON = TRAD / "results.json"
DATA = Path(r"D:\stage2\data\test_images")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def mae_chart(df: pd.DataFrame) -> None:
    y = df["yolo_diff"].abs().mean()
    v = df["vlm_diff"].abs().mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["YOLO26", "GLM-4V-Flash"], [y, v], color=["#4c72b0", "#c44e52"])
    for b, val in zip(bars, [y, v]):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center")
    ax.set_ylabel("平均绝对误差 MAE")
    ax.set_title("两种方案的计数误差")
    ax.set_ylim(0, max(y, v) * 1.3)
    fig.savefig(FIG / "mae_comparison.png", dpi=150)
    plt.close(fig)


def error_hist(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = range(-6, 7)
    ax.hist(df["yolo_diff"], bins=bins, alpha=0.6, label="YOLO26")
    ax.hist(df["vlm_diff"], bins=bins, alpha=0.6, label="GLM-4V-Flash")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("预测数 - 真值")
    ax.set_ylabel("图片数")
    ax.set_title("误差分布")
    ax.legend()
    fig.savefig(FIG / "error_distribution.png", dpi=150)
    plt.close(fig)


def examples() -> None:
    gt = pd.read_csv(GT)
    yolo = json.load(open(YOLO_JSON, encoding="utf-8"))
    picks = []
    for target in [3, 4, 6, 8]:
        sub = gt[gt["total"] == target]
        if not sub.empty:
            picks.append(sub.iloc[0]["filename"])
    picks = picks[:4]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, name in zip(axes.flat, picks):
        p = TRAD / f"{Path(name).stem}_detect.jpg"
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) if p.exists() \
            else cv2.cvtColor(cv2.imread(str(DATA / name)), cv2.COLOR_BGR2RGB)
        g = int(gt[gt.filename == name]["total"].iloc[0])
        y = sum(yolo.get(name, {}).get("counts", {}).values())
        ax.imshow(img)
        ax.set_title(f"真值 {g} 辆 / YOLO 检出 {y} 辆", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(FIG / "detection_examples.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CMP)
    mae_chart(df)
    error_hist(df)
    examples()
    print("saved:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
