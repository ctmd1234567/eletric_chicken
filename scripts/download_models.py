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

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENDPOINT = "https://hf-mirror.com"

# HF 权重摘要来自官方 LFS oid；LwST-FSD 的作者网盘实取摘要单独标注。
MANIFEST = [
    {
        "group": "lwst_fsd",
        "repo": "WCUSTC/LwST-FSD",
        "revision": "af3ed8ac9686a607ec2330275fbb95cf8d1f910d",
        "file": "epoch_10.pth",
        "provider": "ustc_share",
        "share_id": "5b015f2c73e24fd48b82",
        "size": 141991741,
        # 作者公开分享文件的本次实取哈希；上游未提供独立签名/摘要。
        "sha256": "6ac5ca325d59f1bb48a8f63750ba1efd49b53cc5b34659395e8edf14ad8b3add",
        "dest": "models/fire_smoke/lwst_fsd/epoch_10.pth",
    },
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
    {
        # 保留旧组名和本地路径以兼容已有入口；现为 Tenacious Tiger v8.3.0，
        # 不再是 Sensitive Detector v1.1.0。新文件校验成功后才替换旧文件。
        "group": "pyronear_sensitive",
        "repo": "pyronear/yolov11s",
        "revision": "de398b5192b2c1d5f8963aef56bf8df1f13c159e",
        "file": "best.pt",
        "size": 19225626,
        "sha256": "c4b5b846b957cef2465c4f02f657442ec05151b1c3f95cb2e5702c2aa25bc848",
        "dest": "models/fire_smoke/pyronear_sensitive/best.pt",
    },
    {
        "group": "dfire_yolov8n",
        "repo": "rabahdev/fire-smoke-yolov8n",
        "revision": "13017fe8af477c25f5298d168e2dfede4b000753",
        "file": "best.pt",
        "size": 6229802,
        "sha256": "b91633799ceb052c814b4f8b77a37efc9a40f002d528df97d74463585fa4f28f",
        "dest": "models/fire_smoke/dfire_yolov8n/best.pt",
    },
    {
        # FASDD_CV 微调的 YOLO26n；上游旧地址 fire_smoke_detection_box 已重定向。
        "group": "fasdd_yolo26n",
        "repo": "seawsurf/fire_yolo",
        "revision": "9403f231bab33e0ea4048424ff64023936957ba5",
        "file": "weights/best.pt",
        "size": 5377541,
        "sha256": "d2b0be214f45fa01d2044b960c3cb4da4bf65c06a89cf0a1967b0d40e8db4b98",
        "dest": "models/fire_smoke/fasdd_yolo26n/best.pt",
    },
    {
        # Home Fire Dataset 微调；仓库未声明权重许可，仅作为本地待核实候选。
        "group": "safety_yolo26",
        "repo": "e1250/safety_detection",
        "revision": "f2049aa749dcc844e406d2507ffa3b683ce83aac",
        "file": "yolo_smoke_fire.pt",
        "size": 15735298,
        "sha256": "cd2c8c1108b29589d77d4a21cc6e5f111c7586b225e2d8668fda50d791a2cbea",
        "dest": "models/fire_smoke/safety_yolo26/yolo_smoke_fire.pt",
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
    ap.add_argument("--only", choices=sorted({e["group"] for e in MANIFEST}), help="只处理某一组")
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
