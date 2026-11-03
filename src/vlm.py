"""调用智谱 GLM-4V 清点车辆。"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

API_KEY = os.environ.get("VLM_API_KEY", "")
BASE_URL = os.environ.get("VLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4").rstrip("/")
MODEL = os.environ.get("VLM_MODEL", "glm-4v-flash")

PROMPT = """你是车辆清点助手。请清点图中所有机动车（car、bus、truck、motorcycle），
不要统计非机动车和行人。只输出 JSON，格式如下，不要输出其他内容：
{"cars": 数字, "buses": 数字, "trucks": 数字, "motorcycles": 数字, "total": 数字, "description": "一句话描述场景"}"""


def encode_image(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def call_vlm(image_path: str | Path, prompt: str = PROMPT, model: str = MODEL,
             timeout: int = 90, retries: int = 2):
    """返回 (原始文本, 耗时秒, usage)。"""
    img_b64 = encode_image(image_path)
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": prompt},
            ]}
        ],
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    last_err = None
    for _ in range(retries + 1):
        t0 = time.time()
        try:
            r = requests.post(f"{BASE_URL}/chat/completions", json=payload, headers=headers, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            return content, time.time() - t0, data.get("usage", {})
        except Exception as e:
            last_err = e
            time.sleep(2)
    raise RuntimeError(f"VLM 调用失败: {last_err}")


def parse_json(text: str):
    """从模型输出里抠出 JSON。"""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    content, dt, usage = call_vlm(args.image, model=args.model)
    print("原始返回:", content)
    print(f"耗时: {dt:.2f}s  usage: {usage}")
    print("解析结果:", parse_json(content))


if __name__ == "__main__":
    main()
