"""图片和视频按帧读取；不把整段视频放入内存。"""
from contextlib import contextmanager
from dataclasses import dataclass
import math
import time

import cv2

from sentinel.models.weights import sha256_of

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.m4v', '.wmv'}


def source_paths(source, *, video_only=False):
    extensions = VIDEO_EXTENSIONS if video_only else VIDEO_EXTENSIONS | IMAGE_EXTENSIONS
    paths = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in extensions) if source.is_dir() else [source]
    if not paths or any(not p.is_file() or p.suffix.lower() not in extensions for p in paths):
        raise ValueError(f'没有可读取的素材: {source}')
    return paths


@contextmanager
def video_capture(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f'无法打开视频: {path}')
        yield cap
    finally:
        cap.release()


def inspect_video(path):
    with video_capture(path) as cap:
        fps = cap.get(cv2.CAP_PROP_FPS)
        ok, frame = cap.read()
        if not ok or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f'无法解码视频或 FPS 无效: {path.name}')
        return dict(video_id=path.name, sha256=sha256_of(path), fps=fps,
                    declared_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    width=frame.shape[1], height=frame.shape[0])


@dataclass
class Frame:
    index: int
    image: object
    read_ms: float


def video_frames(path, stride, info):
    with video_capture(path) as cap:
        index = 0
        while True:
            start = time.perf_counter()
            ok, image = cap.read()
            read_ms = (time.perf_counter() - start) * 1000
            if not ok:
                break
            if index % stride == 0:
                yield Frame(index, image, read_ms)
            index += 1
        if info['declared_frames'] > 0 and index != info['declared_frames']:
            raise ValueError(f"视频声明 {info['declared_frames']} 帧，实际解码 {index} 帧: {path.name}")
        info['decoded_frames'] = index


def image_frame(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f'无法读取图片: {path}')
    return Frame(0, image, 0)
