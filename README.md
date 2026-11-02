# 停车场车辆清点与方案对比

同一个视觉问题——**给一张停车场/路边照片，清点里面的车辆**——用两套方案实现并对比：

- **传统方案**：YOLO 检测车辆 + SAM 分割每辆车
- **大模型方案**：调用多模态大模型 API，让它清点车辆并描述场景

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
├── data/
│   ├── test_images/    # 测试图（不入库）
│   └── ground_truth.csv
├── assets/             # 测试用的小图
├── weights/            # 模型权重（不入库）
├── results/            # 输出（不入库）
├── figures/            # 对比图
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
# 检测
D:\stage2\venv\Scripts\python.exe -m src.detect --source assets

# 分割（给一个框）
D:\stage2\venv\Scripts\python.exe -m src.segment --image assets/demo1.jpg --box 300 100 1800 1350
```
