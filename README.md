# 道路车辆清点与方案对比

同一张道路/街景图片，用两套方案清点车辆并对比：

- **传统方案**：YOLO26 检测车辆，可选 SAM 分割
- **大模型方案**：智谱 GLM-4V-Flash 清点并描述场景

## 数据

从 COCO val2017 里选 40 张道路场景图，每张 3～8 辆车，车辆是画面主体。真值数量来自 COCO 官方标注（car / bus / truck / motorcycle），不用人工数。

```powershell
python scripts/prepare_coco.py
```

- 图片：`data/test_images/`
- 真值表：`data/ground_truth.csv`

## 目录

```
stage2/
├── src/
│   ├── pipeline_traditional.py   # 传统方案：YOLO 检测 + 可选 SAM 分割
│   ├── pipeline_vlm.py           # 大模型方案：批量调用
│   ├── vlm.py                    # 智谱 API 封装
│   ├── segment.py                # SAM 分割
│   ├── evaluate.py               # 对比指标
│   └── compare.py                # 并排对比图
├── scripts/
│   ├── prepare_coco.py           # 准备数据
│   └── demo_yolo_sam.py          # 检测 + 分割演示
├── data/
│   ├── test_images/              # 测试图，不入库
│   └── ground_truth.csv
├── weights/                      # 模型权重，不入库
├── results/                      # 输出，不入库
├── figures/
└── docs/
```

## 环境

```powershell
uv venv D:\stage2\venv --python 3.12
uv pip install --python D:\stage2\venv\Scripts\python.exe -r requirements.txt
```

权重放在 `D:\stage2\weights\`：

- `yolo26n.pt`：YOLO26 检测权重
- `sam_vit_b_01ec64.pth`：SAM 分割权重

检测用的是 COCO 预训练权重直接推理，没有微调，所以不需要 `data.yaml`。

## 运行

```powershell
# 1. 准备数据
python scripts/prepare_coco.py

# 2. 传统方案（加 --sam 启用分割）
python -m src.pipeline_traditional --sam

# 3. 大模型方案
python -m src.pipeline_vlm

# 4. 指标和对比图
python -m src.evaluate
python -m src.compare
```

大模型用的 key 放在 `.env`：

```
VLM_API_KEY=...
VLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
VLM_MODEL=glm-4v-flash
```

## 结果

| 方案 | MAE | 完全一致 | 单张耗时 | 成本 |
|---|---|---|---|---|
| YOLO26 | 1.25 | 12/40 | 约 0.05s | 0 |
| GLM-4V-Flash | 1.10 | 9/40 | 2.11s | 537 tokens/张 |

YOLO 快、能给出位置框；大模型平均误差略低，能描述场景。对比图见 `figures/comparison_samples.png`，详细分析见 `docs/对比实验报告.md`。
