#!/usr/bin/env python
"""从腾讯视频（v.qq.com）下载单条视频到本地，仅用于个人分析/测试素材。

用法:
    python scripts/fetch_qqvideo.py https://v.qq.com/x/page/s3248s55a3f.html
    python scripts/fetch_qqvideo.py s3248s55a3f -o video

做法:
    1. 从链接里取出 vid（/x/page/<vid>.html）；
    2. 请求播放器自己用的 getinfo 接口，拿到流清单、时长、vkey；
    3. 若返回的流带 DRM（drm != 0）则直接退出，不做任何解密；
    4. 用 直链前缀 + 文件名 + vkey 下载 mp4。

限制与声明:
    - 只处理**非 DRM** 的公开流，遇到加密/付费/需登录的内容会明确报错退出。
    - 下载内容的版权归原权利人，仅供本地分析，不要二次分发或提交到公开仓库。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from sentinel.paths import ROOT
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
GETINFO = "https://vv.video.qq.com/getinfo"


def parse_vid(arg: str) -> str:
    m = re.search(r"/x/(?:page|cover)/([0-9a-zA-Z]{8,})", arg)
    if m:
        return m.group(1)
    m = re.search(r"\b([a-z0-9]{10,})\b", arg)
    if m:
        return m.group(1)
    raise SystemExit(f"没能从 {arg!r} 里解析出 vid")


def get_info(vid: str, fmt: str | None = None) -> dict:
    params = {"vids": vid, "platform": "101001", "charge": "0", "otype": "json"}
    if fmt:
        params["defn"] = fmt
    url = GETINFO + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": "https://v.qq.com/"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read().decode("utf-8", errors="replace")
    raw = re.sub(r"^QZOutputJson=", "", raw).strip().rstrip(";")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"getinfo 返回的不是合法 JSON: {exc}")


def pick_stream(info: dict):
    vi = (info.get("vl") or {}).get("vi") or []
    if not vi:
        raise SystemExit("流清单为空：视频可能已下架、需要登录或为付费内容")
    vi.sort(key=lambda v: (v.get("br") or 0, v.get("fs") or 0), reverse=True)
    return vi[0]


def build_url(stream: dict) -> str:
    ui = ((stream.get("ul") or {}).get("ui") or [])
    if not ui:
        raise SystemExit("没有可用的下载地址前缀")
    base = ui[0]["url"]
    if base.startswith("http://"):
        base = "https://" + base[len("http://"):]
    return f"{base}{stream['fn']}?vkey={stream['fvkey']}"


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": "https://v.qq.com/"})
    with urllib.request.urlopen(req, timeout=120) as r:
        total = int(r.headers.get("Content-Length") or 0)
        done, t0 = 0, time.time()
        with dest.open("wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 / total
                    speed = done / max(time.time() - t0, 1e-6) / 1e6
                    print(f"\r    {done/1e6:6.2f}/{total/1e6:6.2f} MB {pct:5.1f}% {speed:5.1f} MB/s", end="")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="视频链接或 vid")
    ap.add_argument("-o", "--out", default="video", help="输出目录（默认 ./video）")
    ap.add_argument("--name", help="自定义文件名（不含扩展名）")
    args = ap.parse_args()

    vid = parse_vid(args.target)
    outdir = Path(args.out) if Path(args.out).is_absolute() else (ROOT / args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] 查询 {vid}")
    info = get_info(vid)
    if info.get("em") not in (0, None):
        raise SystemExit(f"接口返回错误 em={info.get('em')} msg={info.get('msg')}")

    stream = pick_stream(info)
    drm = stream.get("drm") or 0
    dur = stream.get("td")
    w, h = stream.get("vw"), stream.get("vh")
    print(f"      时长 {dur}s  分辨率 {w}x{h}  码率 {stream.get('br')}  大小 {stream.get('fs')} B")
    if drm:
        raise SystemExit(
            f"该流受 DRM 保护（drm={drm}），不做解密，已停止。请改用其他合法来源。"
        )

    print("[2/3] 下载")
    dest = outdir / f"{(args.name or vid)}.mp4"
    download(build_url(stream), dest)

    print(f"[3/3] 完成: {dest}  ({dest.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
