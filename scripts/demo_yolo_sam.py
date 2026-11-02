"""演示：YOLO 检出车辆 -> 把检测框作为 SAM 提示 -> 输出每辆车的掩膜。

用法：
    python scripts/demo_yolo_sam.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO

WEIGHTS_YOLO = Path(r"D:\stage2\weights\yolo11n.pt")
WEIGHTS_SAM = Path(r"D:\stage2\weights\sam_vit_b_01ec64.pth")
DATA = Path(r"D:\stage2\data\test_images")
OUT = Path(r"D:\stage2\results\demo_yolo_sam.png")
VEH = {"car", "bus", "truck", "motorcycle"}


def main() -> None:
    img_path = sorted(DATA.glob("*.jpg"))[0]
    yolo = YOLO(str(WEIGHTS_YOLO))
    res = yolo(str(img_path), conf=0.25, verbose=False)[0]

    boxes, labels = [], []
    for i in range(len(res.boxes)):
        name = yolo.names[int(res.boxes.cls[i])]
        if name not in VEH:
            continue
        boxes.append(res.boxes.xyxy[i].cpu().numpy())
        labels.append(name)

    print(f"{img_path.name}: 检出 {len(boxes)} 辆车 {labels}")

    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    sam = sam_model_registry["vit_b"](checkpoint=str(WEIGHTS_SAM))
    predictor = SamPredictor(sam)
    predictor.set_image(img)

    overlay = img.copy()
    colors = [(255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 200, 0), (0, 200, 200)]
    for k, box in enumerate(boxes):
        masks, scores, _ = predictor.predict(box=box, multimask_output=False)
        m = masks[0]
        color = np.array(colors[k % len(colors)])
        overlay[m] = (overlay[m] * 0.5 + color * 0.5).astype(np.uint8)
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), tuple(int(c) for c in color), 2)
        cv2.putText(overlay, f"{labels[k]} {float(scores[0]):.2f}", (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, tuple(int(c) for c in color), 2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
