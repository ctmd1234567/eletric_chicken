#!/usr/bin/env python
"""从登记的上游来源下载候选烟火检测权重，并做 SHA-256 校验。

用法:
    python scripts/download_models.py                  # 默认走 hf-mirror.com 镜像
    python scripts/download_models.py --endpoint https://huggingface.co
    python scripts/download_models.py --only firesmockdetect
    python scripts/download_models.py --check          # 只校验已有文件，不下载

说明:
    - 默认端点用 hf-mirror.com，因为国内直连 huggingface.co 常常不通；
      Hugging Face 镜像与官方同源，下载后用 SHA-256 与官方 LFS oid 比对，
      校验通过即可认为与官方文件逐字节一致。LwST-FSD 另从作者公开网盘下载，
      固定本次实取摘要；其上游没有独立摘要可供交叉核对。
    - 不同候选的许可声明不完全相同，部分尚不明确；下载行为不等于获得
      再分发或商用授权，逐项说明见 NOTICE.md。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

DEFAULT_ENDPOINT = "https://hf-mirror.com"

from sentinel.paths import ROOT
from .catalog import MODELS


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
    if entry.get('provider') == 'ustc_share':
        api = f"https://pan.ustc.edu.cn/api/v1/share/get_share_detail?share_id={entry['share_id']}&path=/"
        with urllib.request.urlopen(api, timeout=30) as response:
            detail = json.load(response)
        data = detail.get('data', {})
        if detail.get('code') != 1 or not data.get('permissions', {}).get('can_download'):
            raise RuntimeError('作者网盘分享不可下载，请核对上游公开地址')
        url = data['download_link']
        print(f"  来源: https://pan.ustc.edu.cn/share/index/{entry['share_id']}")
    else:
        url = f"{endpoint.rstrip('/')}/{entry['repo']}/resolve/{entry['revision']}/{entry['file']}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  下载 {entry['file']}")
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
    ap.add_argument("--only", choices=sorted({e["group"] for e in MODELS.values()}), help="只处理某一组")
    ap.add_argument("--check", action="store_true", help="只校验，不下载")
    ap.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = ap.parse_args()

    entries = [e for e in MODELS.values() if not args.only or e["group"] == args.only]
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
