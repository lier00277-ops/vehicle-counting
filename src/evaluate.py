"""对比传统方案和大模型方案的计数误差：总体、每类、漏检/多检。"""
from __future__ import annotations

import argparse
import json

import pandas as pd

CLASSES = ("cars", "buses", "trucks", "motorcycles")
YOLO_KEY = {"cars": "car", "buses": "bus", "trucks": "truck", "motorcycles": "motorcycle"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=r"D:\stage2\data\ground_truth.csv")
    ap.add_argument("--traditional", default=r"D:\stage2\results\traditional\results.json")
    ap.add_argument("--vlm", default=r"D:\stage2\results\vlm_results.json")
    ap.add_argument("--out", default=r"D:\stage2\results\comparison.csv")
    args = ap.parse_args()

    gt = pd.read_csv(args.gt)
    trad = json.load(open(args.traditional, encoding="utf-8"))
    vlm = json.load(open(args.vlm, encoding="utf-8"))

    rows = []
    for _, r in gt.iterrows():
        name = r["filename"]
        ycounts = trad.get(name, {}).get("counts", {})
        parsed = vlm.get(name, {}).get("parsed") or {}
        row = dict(
            file=name,
            gt=int(r["total"]),
            yolo=sum(ycounts.values()),
            vlm=int(parsed.get("total", 0)) if parsed else None,
        )
        for c in CLASSES:
            row[f"gt_{c}"] = int(r[c])
            row[f"yolo_{c}"] = int(ycounts.get(YOLO_KEY[c], 0))
            row[f"vlm_{c}"] = int(parsed.get(c, 0)) if parsed else None
        row["yolo_diff"] = row["yolo"] - row["gt"]
        row["vlm_diff"] = (row["vlm"] - row["gt"]) if row["vlm"] is not None else None
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(df[["file", "gt", "yolo", "vlm", "yolo_diff", "vlm_diff"]].to_string(index=False))

    print("\n== 总体 ==")
    print(f"样本数: {len(df)}")
    print(f"YOLO  MAE: {df['yolo_diff'].abs().mean():.2f}   完全一致: {(df['yolo_diff'] == 0).sum()}/{len(df)}")
    print(f"VLM   MAE: {df['vlm_diff'].abs().mean():.2f}   完全一致: {(df['vlm_diff'] == 0).sum()}/{len(df)}")

    for name, col in [("YOLO", "yolo_diff"), ("VLM", "vlm_diff")]:
        d = df[col]
        print(f"{name}: 漏检 {int(d[d < 0].abs().sum())} 辆（涉及 {int((d < 0).sum())} 张图），"
              f"多检 {int(d[d > 0].sum())} 辆（涉及 {int((d > 0).sum())} 张图）")

    print("\n== 每类误差（MAE）==")
    print(f"{'类别':<14}{'YOLO':>8}{'VLM':>8}")
    for c in CLASSES:
        y = (df[f"yolo_{c}"] - df[f"gt_{c}"]).abs().mean()
        v = (df[f"vlm_{c}"] - df[f"gt_{c}"]).abs().mean()
        print(f"{c:<14}{y:>8.2f}{v:>8.2f}")

    lat = [x.get("latency") for x in vlm.values() if x.get("latency")]
    if lat:
        print(f"\nVLM 平均耗时: {sum(lat) / len(lat):.2f}s/张（YOLO 约 0.05s/张）")


if __name__ == "__main__":
    main()
