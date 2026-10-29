#!/usr/bin/env python
"""用 BeautifulSoup + urllib 从 B 站页面里取播放地址，下载单个视频到本地。

用法:
    python scripts/fetch_bilibili.py BV1xPe8zfEoh
    python scripts/fetch_bilibili.py https://www.bilibili.com/video/BV1xPe8zfEoh/ -o video
    python scripts/fetch_bilibili.py BV1xPe8zfEoh --list-quality

做法（不依赖 yt-dlp / requests，只用标准库 + bs4）:
    1. 带浏览器 UA / Referer 请求视频页，顺带拿到 buvid 之类的基础 Cookie；
    2. BeautifulSoup 解析页面里的 <script>window.__INITIAL_STATE__=…</script>，
       取出 bvid / aid / cid / 标题；
    3. 请求 x/player/playurl 拿 DASH 清单，分离出视频流和音频流；
    4. 分别下载 m4s，再用 ffmpeg 无损封装成一个 mp4。

仅用于本地分析/测试素材。下载内容的版权归原作者，请勿二次分发或提交到公开仓库。
"""

from __future__ import annotations

import argparse
import gzip
import http.cookiejar
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
FFMPEG_FALLBACK = os.environ.get("FFMPEG_PATH")  # 可选：ffmpeg 不在 PATH 时用环境变量指定

QUALITY = {127: "8K", 126: "杜比视界", 125: "HDR", 120: "4K", 116: "1080P60", 112: "1080P+", 80: "1080P", 74: "720P60", 64: "720P", 32: "480P", 16: "360P"}


def build_opener() -> urllib.request.OpenerDirector:
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.addheaders = [
        ("User-Agent", UA),
        ("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"),
        ("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8"),
        ("Accept-Encoding", "gzip, deflate"),
    ]
    return opener


def _decompress(raw: bytes, encoding: str) -> bytes:
    """B 站会返回 gzip/deflate 压缩体，urllib 不会自动解压。"""
    encoding = (encoding or "").lower()
    if "gzip" in encoding:
        return gzip.decompress(raw)
    if "deflate" in encoding:
        try:
            return zlib.decompress(raw)
        except zlib.error:
            return zlib.decompress(raw, -zlib.MAX_WBITS)
    return raw


def get_text(opener, url: str, referer: str | None = None) -> str:
    req = urllib.request.Request(url)
    if referer:
        req.add_header("Referer", referer)
    with opener.open(req, timeout=30) as r:
        raw = _decompress(r.read(), r.headers.get("Content-Encoding", ""))
    return raw.decode("utf-8", errors="replace")


def get_json(opener, url: str, referer: str) -> dict:
    return json.loads(get_text(opener, url, referer))


def parse_bvid(arg: str) -> str:
    m = re.search(r"(BV[0-9A-Za-z]{10})", arg)
    if not m:
        raise SystemExit(f"没能从 {arg!r} 里解析出 BV 号")
    return m.group(1)


def page_meta(opener, bvid: str) -> dict:
    url = f"https://www.bilibili.com/video/{bvid}/"
    html = get_text(opener, url)
    soup = BeautifulSoup(html, "html.parser")

    title = None
    if soup.title and soup.title.string:
        title = soup.title.string.replace("_哔哩哔哩_bilibili", "").strip()

    state = None
    for tag in soup.find_all("script"):
        text = tag.string or ""
        if "__INITIAL_STATE__" in text:
            raw = text.split("window.__INITIAL_STATE__=", 1)[1].strip().rstrip(";")
            raw = re.sub(r"\bundefined\b", "null", raw)  # 页面混入 JS 的 undefined
            try:
                # 后面还跟着别的 JS 语句，用 raw_decode 只取第一个完整 JSON 对象
                state, _ = json.JSONDecoder().raw_decode(raw)
            except json.JSONDecodeError as exc:
                print(f"[警告] __INITIAL_STATE__ 解析失败: {exc}")
            break

    if not state:
        raise SystemExit("页面里没找到 __INITIAL_STATE__，可能需要登录 Cookie")

    vd = state.get("videoData") or {}
    pages = vd.get("pages") or []
    return {
        "bvid": vd.get("bvid") or bvid,
        "aid": vd.get("aid"),
        "cid": pages[0].get("cid") if pages else None,
        "title": vd.get("title") or title or bvid,
        "duration": vd.get("duration"),
        "pages": [(p.get("cid"), p.get("part")) for p in pages],
    }


def playurl(opener, bvid: str, cid: int, qn: int = 80) -> dict:
    params = urllib.parse.urlencode({"bvid": bvid, "cid": cid, "qn": qn, "fnval": 4048, "fourk": 1})
    url = "https://api.bilibili.com/x/player/playurl?" + params
    data = get_json(opener, url, referer=f"https://www.bilibili.com/video/{bvid}/")
    if data.get("code") != 0:
        raise SystemExit(f"playurl 接口返回错误 code={data.get('code')} msg={data.get('message')}")
    return data["data"]


def download(opener, url: str, dest: Path, referer: str) -> None:
    req = urllib.request.Request(url)
    req.add_header("Referer", referer)
    req.add_header("User-Agent", UA)
    req.add_header("Accept-Encoding", "identity")  # 二进制流不要压缩，否则存下来的 m4s 是坏的
    with opener.open(req, timeout=180) as r:
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
                    print(f"\r    {done/1e6:6.1f}/{total/1e6:6.1f} MB {pct:5.1f}% {speed:5.1f} MB/s", end="")
    print()


def find_ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    if FFMPEG_FALLBACK and Path(FFMPEG_FALLBACK).exists():
        return FFMPEG_FALLBACK
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="BV 号或视频链接")
    ap.add_argument("-o", "--out", default="video", help="输出目录（默认 ./video）")
    ap.add_argument("--qn", type=int, default=80, help="清晰度请求值，80=1080P")
    ap.add_argument("--list-quality", action="store_true", help="只列出可用清晰度")
    ap.add_argument("--keep-parts", action="store_true", help="保留分离的 m4s 文件")
    args = ap.parse_args()

    bvid = parse_bvid(args.target)
    outdir = Path(args.out) if Path(args.out).is_absolute() else (ROOT / args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    opener = build_opener()
    print(f"[1/4] 解析页面 {bvid}")
    meta = page_meta(opener, bvid)
    print(f"      标题: {meta['title']}")
    print(f"      时长: {meta['duration']} 秒, 分P: {len(meta['pages'])}")
    if not meta["cid"]:
        raise SystemExit("没取到 cid")

    print("[2/4] 请求播放地址")
    data = playurl(opener, bvid, meta["cid"], args.qn)
    dash = data.get("dash")
    if not dash:
        raise SystemExit("该视频没有 DASH 流（可能是番剧/付费/需登录内容）")

    videos = sorted(dash.get("video") or [], key=lambda v: (v.get("id", 0), v.get("bandwidth", 0)), reverse=True)
    audios = sorted(dash.get("audio") or [], key=lambda a: a.get("bandwidth", 0), reverse=True)
    if not videos:
        raise SystemExit("DASH 清单里没有视频流")

    if args.list_quality:
        print("      可用视频流:")
        for v in videos:
            print(f"        qn={v.get('id'):>3} {QUALITY.get(v.get('id'), '?'):<8} {v.get('width')}x{v.get('height')} {v.get('codecs')}")
        return 0

    vs, aus = videos[0], (audios[0] if audios else None)
    print(f"      选中视频流: qn={vs.get('id')} {QUALITY.get(vs.get('id'),'')} {vs.get('width')}x{vs.get('height')} {vs.get('codecs')}")
    if aus:
        print(f"      选中音频流: {aus.get('codecs')} {aus.get('bandwidth')} bps")

    safe = re.sub(r'[\\/:*?"<>|]', "_", meta["title"])[:80].strip() or bvid
    referer = f"https://www.bilibili.com/video/{bvid}/"
    vpath = outdir / f"{safe}.video.m4s"
    apath = outdir / f"{safe}.audio.m4s"

    print("[3/4] 下载视频流")
    download(opener, vs["baseUrl"], vpath, referer)
    if aus:
        print("      下载音频流")
        download(opener, aus["baseUrl"], apath, referer)

    print("[4/4] 封装")
    final = outdir / f"{safe}.mp4"
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print(f"      没找到 ffmpeg，原始分片保留在 {vpath}")
        return 0

    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(vpath)]
    if aus:
        cmd += ["-i", str(apath)]
    cmd += ["-c", "copy", str(final)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"      ffmpeg 封装失败: {res.stderr[:500]}")
        return 1
    print(f"      完成: {final}  ({final.stat().st_size/1e6:.1f} MB)")
    if not args.keep_parts:
        vpath.unlink(missing_ok=True)
        apath.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
