"""批量调用大模型，结果缓存到 JSON。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.vlm import call_vlm, parse_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=r"D:\stage2\data\test_images")
    ap.add_argument("--out", default=r"D:\stage2\results\vlm_results.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}

    imgs = sorted(Path(args.source).glob("*.jpg"))
    if args.limit:
        imgs = imgs[: args.limit]
    for i, p in enumerate(imgs, 1):
        if p.name in results and results[p.name].get("parsed"):
            print(f"[{i}/{len(imgs)}] {p.name} 已有缓存")
            continue
        try:
            text, dt, usage = call_vlm(p)
            results[p.name] = {"raw": text, "parsed": parse_json(text),
                               "latency": round(dt, 2), "usage": usage}
            print(f"[{i}/{len(imgs)}] {p.name} -> {results[p.name]['parsed']} ({dt:.1f}s)")
        except Exception as e:  # noqa: BLE001
            results[p.name] = {"error": str(e)}
            print(f"[{i}/{len(imgs)}] {p.name} 失败: {e}")

    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved ->", out)


if __name__ == "__main__":
    main()
