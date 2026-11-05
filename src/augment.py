"""用 OpenCV 做数据增强，生成对比图。"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(r"D:\stage2\data\test_images")
OUT = Path(r"D:\stage2\figures\augmentation_demo.png")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    img = cv2.imread(str(sorted(DATA.glob("*.jpg"))[0]))
    h, w = img.shape[:2]

    flip = cv2.flip(img, 1)                                     # 水平翻转
    crop = img[int(h * 0.15):int(h * 0.85), int(w * 0.1):int(w * 0.9)]  # 裁剪
    bright = cv2.convertScaleAbs(img, alpha=1.15, beta=30)      # 亮度
    small = cv2.resize(img, (int(w * 0.6), int(h * 0.6)))
    up = cv2.resize(small, (w, h))                              # 缩小再放大
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    items = [("原图", img), ("水平翻转", flip), ("随机裁剪", crop),
             ("亮度调整", bright), ("缩小再放大", up), ("灰度", gray)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (title, im) in zip(axes.flat, items):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
