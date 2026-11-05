"""在 Kaggle GPU 上微调 YOLO26 车辆检测模型。

用法：把本文件内容粘贴到 Kaggle Notebook 的一个单元格里运行。
前提：
1. 已上传 vehicle_det_kaggle.zip 为 Kaggle Dataset
2. Notebook 右侧 Accelerator 选 GPU，Internet 选 On（装 ultralytics 用）
"""
import glob
import os
import shutil
import zipfile

from ultralytics import YOLO

# 1. 找到数据集压缩包并解压到可写目录
out = "/kaggle/working/data"
zipp = glob.glob("/kaggle/input/**/*.zip", recursive=True)
if zipp:
    print("压缩包:", zipp[0])
    shutil.rmtree(out, ignore_errors=True)
    with zipfile.ZipFile(zipp[0]) as z:
        z.extractall(out)
else:
    # 如果 Kaggle 已经解压，就直接找 yolo26n.pt 所在目录
    hits = glob.glob("/kaggle/input/**/yolo26n.pt", recursive=True)
    assert hits, "没找到 yolo26n.pt，请检查数据集"
    out = os.path.dirname(hits[0])

yolo_w = os.path.join(out, "yolo26n.pt")
data_dir = os.path.join(out, "vehicle_det")
assert os.path.exists(yolo_w), f"没找到权重: {yolo_w}"
assert os.path.isdir(os.path.join(data_dir, "images", "train")), f"数据集结构不对: {data_dir}"
print("权重:", yolo_w)
print("数据:", data_dir)

# 2. 生成 data.yaml
YAML = "/kaggle/working/vehicle.yaml"
with open(YAML, "w", encoding="utf-8") as f:
    f.write(
        f"path: {data_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: car\n"
        "  1: truck\n"
        "  2: bus\n"
        "  3: motorcycle\n"
    )

# 3. 微调
model = YOLO(yolo_w)
model.train(
    data=YAML,
    epochs=30,
    imgsz=640,
    batch=16,
    device=0,
    fliplr=0.5,
    mosaic=1.0,
    project="/kaggle/working/runs",
    name="vehicle",
    exist_ok=True,
)

# 4. 验证并导出结果
metrics = model.val(data=YAML)
print("mAP50:", round(float(metrics.box.map50), 4))
print("mAP50-95:", round(float(metrics.box.map), 4))

run = "/kaggle/working/runs/vehicle"
for name in ["results.csv", "results.png", "confusion_matrix.png",
             "confusion_matrix_normalized.png", "BoxPR_curve.png", "BoxF1_curve.png"]:
    src = os.path.join(run, name)
    if os.path.exists(src):
        shutil.copy(src, os.path.join("/kaggle/working", name))
best = os.path.join(run, "weights", "best.pt")
if os.path.exists(best):
    shutil.copy(best, "/kaggle/working/vehicle_best.pt")
print("done:", sorted(os.listdir("/kaggle/working")))
