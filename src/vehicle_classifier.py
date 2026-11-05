"""用 torchvision 微调车辆类型分类器（car/truck/bus/motorcycle）。

数据来自 COCO 标注框裁剪出的车辆小图，训练时用随机裁剪和水平翻转做增强。
"""
from __future__ import annotations

import json
import random
import zipfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

DATA = Path(r"D:\stage2\data\test_images")
ZIP = Path(r"D:\stage2\data\coco\annotations_trainval2017.zip")
FIG = Path(r"D:\stage2\figures")
NAMES = ["car", "truck", "bus", "motorcycle"]
CAT = {"car": 0, "truck": 1, "bus": 2, "motorcycle": 3}
DEVICE = "cpu"
SEED = 42

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_samples():
    with zipfile.ZipFile(ZIP) as z, z.open("annotations/instances_val2017.json") as f:
        data = json.load(f)
    cats = {c["id"]: c["name"] for c in data["categories"]}
    files = {im["id"]: im["file_name"] for im in data["images"]}
    wanted = {p.name for p in DATA.glob("*.jpg")}

    samples = []
    for a in data["annotations"]:
        if a.get("iscrowd"):
            continue
        name = cats[a["category_id"]]
        if name not in CAT or files[a["image_id"]] not in wanted:
            continue
        img = cv2.imread(str(DATA / files[a["image_id"]]))
        x, y, w, h = [int(v) for v in a["bbox"]]
        crop = img[max(y, 0):y + h, max(x, 0):x + w]
        if crop.size == 0:
            continue
        samples.append((crop, CAT[name]))
    return samples


class CropDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        crop, label = self.samples[i]
        return self.transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), label


def split(samples):
    by_class: dict[int, list] = {}
    for s in samples:
        by_class.setdefault(s[1], []).append(s)
    train, val = [], []
    rng = random.Random(SEED)
    for items in by_class.values():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * 0.2))
        val += items[:n_val]
        train += items[n_val:]
    return train, val


def main() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)

    samples = load_samples()
    train, val = split(samples)
    counts = {NAMES[c]: sum(1 for _, y in samples if y == c) for c in range(4)}
    print("样本数:", len(samples), counts, "| train", len(train), "val", len(val))

    train_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(112, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(CropDataset(train, train_tf), batch_size=16, shuffle=True)
    val_loader = DataLoader(CropDataset(val, val_tf), batch_size=16, shuffle=False)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(512, 4)
    model.to(DEVICE)

    opt = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    hist = []
    for epoch in range(1, 31):
        model.train()
        loss_sum = correct = total = 0
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * yb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)

        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_correct += (model(xb).argmax(1) == yb).sum().item()
                val_total += yb.size(0)
        hist.append((epoch, loss_sum / total, correct / total, val_correct / val_total))
        if epoch % 5 == 0:
            print(f"epoch {epoch:02d} loss={hist[-1][1]:.3f} train_acc={hist[-1][2]:.3f} "
                  f"val_acc={hist[-1][3]:.3f}")

    # 曲线
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([h[0] for h in hist], [h[1] for h in hist])
    axes[0].set_title("训练损失")
    axes[0].set_xlabel("epoch")
    axes[1].plot([h[0] for h in hist], [h[2] for h in hist], label="train")
    axes[1].plot([h[0] for h in hist], [h[3] for h in hist], label="val")
    axes[1].set_title("分类准确率")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    plt.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "classifier_curves.png", dpi=150)
    plt.close(fig)

    # 混淆矩阵
    cm = np.zeros((4, 4), dtype=int)
    with torch.no_grad():
        for xb, yb in val_loader:
            for t, p in zip(yb.numpy(), model(xb).argmax(1).numpy()):
                cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(NAMES, rotation=30, ha="right")
    ax.set_yticklabels(NAMES)
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    fig.savefig(FIG / "classifier_confusion.png", dpi=150)
    plt.close(fig)

    print(f"最终验证准确率: {hist[-1][3]:.3f}")
    print("saved ->", FIG / "classifier_curves.png", FIG / "classifier_confusion.png")


if __name__ == "__main__":
    main()
