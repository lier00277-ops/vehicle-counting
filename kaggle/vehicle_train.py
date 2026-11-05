"""在 Kaggle GPU 上微调 YOLO26 车辆检测模型。

用法：把本文件内容粘贴到 Kaggle Notebook 的一个单元格里运行。
前提：
1. 已上传 vehicle_det_kaggle.zip 为 Kaggle Dataset（名字假设为 vehicle-det）
2. Notebook 右侧 Accelerator 选 GPU，Internet 可选（权重已包含在数据集里）
"""
import os
import shutil

from ultralytics import YOLO

DATA_DIR = "/kaggle/input/vehicle-det/vehicle_det"
YAML = "/kaggle/working/vehicle.yaml"
YOLO_WEIGHTS = "/kaggle/input/vehicle-det/yolo26n.pt"
RUN_DIR = "/kaggle/working/runs"

with open(YAML, "w", encoding="utf-8") as f:
    f.write(
        f"path: {DATA_DIR}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: car\n"
        "  1: truck\n"
        "  2: bus\n"
        "  3: motorcycle\n"
    )

model = YOLO(YOLO_WEIGHTS)
model.train(
    data=YAML,
    epochs=30,
    imgsz=640,
    batch=16,
    device=0,
    fliplr=0.5,
    mosaic=1.0,
    project=RUN_DIR,
    name="vehicle",
    exist_ok=True,
)

metrics = model.val(data=YAML)
print("mAP50:", round(float(metrics.box.map50), 4))
print("mAP50-95:", round(float(metrics.box.map), 4))

run = os.path.join(RUN_DIR, "vehicle")
for name in ["results.csv", "results.png", "confusion_matrix.png",
             "confusion_matrix_normalized.png", "BoxPR_curve.png", "BoxF1_curve.png"]:
    src = os.path.join(run, name)
    if os.path.exists(src):
        shutil.copy(src, os.path.join("/kaggle/working", name))
best = os.path.join(run, "weights", "best.pt")
if os.path.exists(best):
    shutil.copy(best, "/kaggle/working/vehicle_best.pt")
print("done, outputs:", sorted(os.listdir("/kaggle/working")))
