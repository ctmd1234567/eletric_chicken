#!/usr/bin/env python
"""校验模型权重：SHA-256 + 真实加载 + 可选推理。

用法:
    python scripts/verify_models.py                      # 只做哈希校验 + 加载
    python scripts/verify_models.py --image samples/a.jpg # 附带跑一次推理
    python scripts/verify_models.py --json reports/verify/verify.json

输出内容:
    - 文件大小 / SHA-256 / 是否与官方一致
    - 任务类型、类别名、参数量、训练时的 imgsz
    - 加载是否成功（失败时打印完整异常，不修改权重文件）
    - 可选：单张图推理耗时与检出的框数
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from download_models import MANIFEST, sha256_of  # noqa: E402

# 每个权重官方声明的工作分辨率（来自模型卡）
NATIVE_IMGSZ = {
    "v5bA_y11m_best.pt": 1280,
    "expB_y11s_deployed.pt": 1280,
    "fire_smoke_yolov8.pt": 640,
}


def inspect(path: Path, image: Path | None) -> dict:
    from ultralytics import YOLO

    rec: dict = {"file": str(path.relative_to(ROOT)).replace("\\", "/"), "load": "failed"}
    t0 = time.perf_counter()
    model = YOLO(str(path))
    rec["load"] = "ok"
    rec["load_seconds"] = round(time.perf_counter() - t0, 3)
    rec["task"] = getattr(model, "task", None)
    rec["classes"] = getattr(model, "names", None)
    try:
        rec["params"] = sum(p.numel() for p in model.model.parameters())
    except Exception:
        rec["params"] = None
    # 训练时的 imgsz 记录在 ckpt 的 train_args 里，缺省则用模型卡声明值
    try:
        args = model.ckpt.get("train_args", {}) or {}
        rec["train_imgsz"] = args.get("imgsz")
        rec["train_data"] = args.get("data")
    except Exception:
        rec["train_imgsz"] = None
    rec["native_imgsz"] = NATIVE_IMGSZ.get(path.name)

    if image is not None:
        imgsz = rec["native_imgsz"] or 640
        t1 = time.perf_counter()
        res = model.predict(str(image), imgsz=imgsz, conf=0.10, verbose=False)
        rec["infer_seconds"] = round(time.perf_counter() - t1, 3)
        r = res[0]
        rec["infer_imgsz"] = imgsz
        rec["n_boxes"] = int(len(r.boxes))
        names = r.names
        rec["detections"] = [
            {
                "cls": names[int(c)],
                "conf": round(float(cf), 4),
                "xyxy": [round(float(v), 1) for v in box],
            }
            for box, c, cf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf)
        ]
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, help="可选：用于冒烟推理的单张图片")
    ap.add_argument("--json", type=Path, help="把结果写入 JSON 文件")
    ap.add_argument("--only", help="只检查某个文件名")
    args = ap.parse_args()

    results = []
    for entry in MANIFEST:
        path = ROOT / entry["dest"]
        print(f"\n== {path.name} ==")
        if not path.exists():
            print("  [缺失] 文件不存在，先运行 scripts/download_models.py")
            results.append({"file": entry["dest"], "load": "missing"})
            continue

        digest = sha256_of(path)
        hash_ok = digest == entry["sha256"]
        print(f"  size   : {path.stat().st_size} bytes (期望 {entry['size']})")
        print(f"  sha256 : {digest}")
        print(f"  一致性 : {'与官方一致' if hash_ok else '不一致！文件可能损坏或被替换'}")

        if args.only and args.only != path.name:
            continue
        try:
            rec = inspect(path, args.image)
            rec["sha256"] = digest
            rec["sha256_match"] = hash_ok
            print(f"  加载   : ok ({rec['load_seconds']}s)")
            print(f"  任务   : {rec['task']}")
            print(f"  类别   : {rec['classes']}")
            print(f"  参数量 : {rec['params']:,}" if rec["params"] else "  参数量 : 未知")
            print(f"  imgsz  : 训练记录={rec.get('train_imgsz')} 模型卡={rec['native_imgsz']}")
            if args.image:
                print(f"  推理   : imgsz={rec['infer_imgsz']} 耗时={rec['infer_seconds']}s 检出={rec['n_boxes']} 个框")
                for d in rec.get("detections", [])[:5]:
                    print(f"           {d['cls']} {d['conf']:.3f} {d['xyxy']}")
        except Exception as exc:  # 保留完整栈，方便定位是缺依赖还是结构不兼容
            rec = {
                "file": entry["dest"],
                "sha256": digest,
                "sha256_match": hash_ok,
                "load": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  加载   : 失败 -> {type(exc).__name__}: {exc}")
            traceback.print_exc()
        results.append(rec)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写入 {args.json}")

    bad = [r for r in results if r.get("load") != "ok"]
    print(f"\n汇总：{len(results) - len(bad)}/{len(results)} 个权重加载成功")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
