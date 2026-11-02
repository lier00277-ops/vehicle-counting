# 道路车辆清点与方案对比

同一个视觉问题——**给一张道路/街景照片，清点里面的车辆**——用两套方案实现并对比：

- **传统方案**：YOLO 检测车辆 + SAM 分割每辆车
- **大模型方案**：调用多模态大模型 API，清点车辆并描述场景

## 数据

从 COCO val2017 里挑出的 **40 张道路场景图**，每张 4～10 辆车。真值数量直接来自 COCO 标注（car / bus / truck / motorcycle），不需要人工数。

```powershell
# 下载 COCO 标注（约 240MB）并挑选、下载 40 张图
python scripts/prepare_coco.py
```

- 图片：`data/test_images/`（不入库）
- 标注表：`data/ground_truth.csv`
- COCO 原始标注：`data/coco/`（不入库）

## 目录

```
stage2/
├── src/
│   ├── detect.py       # YOLO 车辆检测
│   ├── segment.py      # SAM 框提示分割
│   ├── pipeline_traditional.py   # 传统方案（检测+分割+计数）
│   ├── vlm.py          # 大模型 API
│   ├── evaluate.py     # 对比指标
│   └── compare.py      # 并排对比图
├── scripts/
│   ├── prepare_coco.py     # 准备数据集
│   └── demo_yolo_sam.py    # YOLO -> SAM 演示
├── data/
│   ├── test_images/    # 测试图（不入库）
│   └── ground_truth.csv
├── assets/
├── weights/            # 模型权重（不入库）
├── results/            # 输出（不入库）
├── figures/
└── docs/
```

## 环境

```powershell
uv venv D:\stage2\venv --python 3.12
uv pip install --python D:\stage2\venv\Scripts\python.exe -r requirements.txt
```

权重放在 `D:\stage2\weights\`：

- `yolo11n.pt`：YOLO 检测权重
- `sam_vit_b_01ec64.pth`：SAM 分割权重

## 运行

```powershell
# 检测（输出计数 + 标注图 + JSON）
D:\stage2\venv\Scripts\python.exe -m src.detect --source data/test_images

# 分割（给一个框）
D:\stage2\venv\Scripts\python.exe -m src.segment --image data/test_images/xxx.jpg --box x1 y1 x2 y2

# YOLO 检测 -> SAM 分割 演示
D:\stage2\venv\Scripts\python.exe scripts/demo_yolo_sam.py
```

## 当前结果

YOLO 基线（40 张测试图）：平均绝对误差 **MAE = 1.52**，完全一致 7/40；平均真值 5.7、平均预测 4.8（略有漏检）。

大模型方案和对比实验待补充。
