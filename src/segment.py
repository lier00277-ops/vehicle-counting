"""SAM 框提示分割：给一个框，输出掩膜。"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

SAM_WEIGHTS = Path(r"D:\stage2\weights\sam_vit_b_01ec64.pth")


def load_predictor(model_type: str = "vit_b", checkpoint: Path = SAM_WEIGHTS) -> SamPredictor:
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    return SamPredictor(sam)


def segment_box(predictor: SamPredictor, image_rgb: np.ndarray, box) -> tuple[np.ndarray, float]:
    """box: [x1, y1, x2, y2] 像素坐标。"""
    predictor.set_image(image_rgb)
    masks, scores, _ = predictor.predict(box=np.array(box), multimask_output=False)
    return masks[0], float(scores[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--box", nargs=4, type=float, required=True, help="x1 y1 x2 y2")
    ap.add_argument("--out", default=r"D:\stage2\results\segment.png")
    args = ap.parse_args()

    img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    predictor = load_predictor()
    mask, score = segment_box(predictor, img, args.box)

    overlay = img.copy()
    overlay[mask] = (overlay[mask] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"score={score:.3f} mask_pixels={int(mask.sum())} saved {args.out}")


if __name__ == "__main__":
    main()
