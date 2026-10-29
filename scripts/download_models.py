#!/usr/bin/env python
"""从 Hugging Face 下载本项目的两个候选烟雾检测模型权重，并做 SHA-256 校验。

用法:
    python scripts/download_models.py                  # 默认走 hf-mirror.com 镜像
    python scripts/download_models.py --endpoint https://huggingface.co
    python scripts/download_models.py --only firesmockdetect
    python scripts/download_models.py --check          # 只校验已有文件，不下载

说明:
    - 默认端点用 hf-mirror.com，因为国内直连 huggingface.co 常常不通；
      镜像与官方是同源的 Git 仓库，下载后用 SHA-256 与官方 LFS oid 比对，
      校验通过即可认为与官方文件逐字节一致。
    - 权重许可证为 AGPL-3.0，下载行为本身不等于获得任何额外授权，
      详见 NOTICE.md。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENDPOINT = "https://hf-mirror.com"

# sha256 取自 Hugging Face API 返回的 LFS oid（即官方文件哈希）
MANIFEST = [
    {
        "group": "firesmockdetect",
        "repo": "TianchenGuan/FireSmockDetect-models",
        "revision": "a914a2a036686cf09bac8efc738971fdc8f31a29",
        "file": "v5bA_y11m_best.pt",
        "size": 40578796,
        "sha256": "1b6cb82295767ae9d0d5c4fb94dffcaf17bbbafc0e3af001d3fe517015902298",
        "dest": "models/fire_smoke/firesmockdetect/v5bA_y11m_best.pt",
    },
    {
        "group": "firesmockdetect",
        "repo": "TianchenGuan/FireSmockDetect-models",
        "revision": "a914a2a036686cf09bac8efc738971fdc8f31a29",
        "file": "expB_y11s_deployed.pt",
        "size": 19241370,
        "sha256": "cc56224c84092e1e374cee1b3ce317f17f7f80e7945bc6c9a3a24a083321ff0e",
        "dest": "models/fire_smoke/firesmockdetect/expB_y11s_deployed.pt",
    },
    {
        "group": "early_fire_smoke",
        "repo": "mfranzon/fire-smoke-yolov8",
        "revision": "f1c6426b069c1849cbf13b1ef5d2a260289286db",
        "file": "fire_smoke_yolov8.pt",
        "size": 6262051,
        "sha256": "ac0a10257b2bc1f20c9d957f8adeeb61dd6140322fc19d0b4a116cb491776d16",
        "dest": "models/fire_smoke/early_fire_smoke/fire_smoke_yolov8.pt",
    },
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, entry: dict) -> bool:
    if not path.exists():
        print(f"  [缺失] {path}")
        return False
    size = path.stat().st_size
    digest = sha256_of(path)
    ok_size = size == entry["size"]
    ok_hash = digest == entry["sha256"]
    status = "OK" if (ok_size and ok_hash) else "不匹配"
    print(f"  [{status}] {path.name}  size={size}  sha256={digest[:16]}…")
    if not ok_size:
        print(f"         期望大小 {entry['size']}，实际 {size}")
    if not ok_hash:
        print(f"         期望 sha256 {entry['sha256']}")
    return ok_size and ok_hash


def download(entry: dict, endpoint: str, force: bool = False) -> bool:
    dest = ROOT / entry["dest"]
    if dest.exists() and not force and verify(dest, entry):
        print("  已存在且校验通过，跳过下载")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{endpoint.rstrip('/')}/{entry['repo']}/resolve/{entry['revision']}/{entry['file']}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  下载 {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "eletric-chicken/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 / total
                print(f"\r    {done/1e6:7.2f} / {total/1e6:7.2f} MB  ({pct:5.1f}%)", end="")
        print()

    if not verify(tmp, entry):
        tmp.unlink(missing_ok=True)
        print("  校验失败，已删除临时文件；请换端点重试（--endpoint https://huggingface.co）")
        return False
    tmp.replace(dest)
    print(f"  已保存到 {dest}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT))
    ap.add_argument("--only", choices=["firesmockdetect", "early_fire_smoke"], help="只处理某一组")
    ap.add_argument("--check", action="store_true", help="只校验，不下载")
    ap.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = ap.parse_args()

    entries = [e for e in MANIFEST if not args.only or e["group"] == args.only]
    failures = 0
    for e in entries:
        print(f"\n== {e['group']} / {e['file']} ==")
        if args.check:
            ok = verify(ROOT / e["dest"], e)
        else:
            ok = download(e, args.endpoint, args.force)
        failures += 0 if ok else 1

    print(f"\n完成：{len(entries) - failures}/{len(entries)} 个文件校验通过")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
