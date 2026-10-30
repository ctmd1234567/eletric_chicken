"""H.264 编码和系统播放器，与模型推理无关。"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


class MP4Writer:
    """通过 FFmpeg 写入 H.264 MP4；编码完成后才生成最终文件。"""

    def __init__(self, output: Path, fps: float, size: tuple[int, int]):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("保存兼容 MP4 需要 FFmpeg，请安装并将 ffmpeg 加入 PATH。")
        self.output = output
        self.temporary = output.with_suffix(".mp4.part")
        self.closed = False
        self.errors = tempfile.TemporaryFile()
        width, height = size
        try:
            self.process = subprocess.Popen(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
                 "-r", str(fps), "-i", "pipe:0", "-an",
                 "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4",
                 str(self.temporary)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.errors,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            self.errors.close()
            raise

    def write(self, frame):
        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            self.release()
            raise RuntimeError(f"FFmpeg 在写入视频时退出: {self.output}") from None

    def release(self):
        if self.closed:
            return
        self.closed = True
        try:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            code = self.process.wait()
            self.errors.seek(0)
            error = self.errors.read().decode("utf-8", errors="replace")
            if code != 0:
                raise RuntimeError(f"MP4 编码失败（退出码 {code}）: {error}")
            if not self.temporary.exists() or self.temporary.stat().st_size == 0:
                raise RuntimeError(f"MP4 编码未生成有效文件: {self.output}")
            self.temporary.replace(self.output)
        finally:
            self.errors.close()


def open_video(path: Path):
    """交给系统播放器播放完整视频，与图片预览分开。"""
    if sys.platform == "win32":
        os.startfile(str(path.resolve()))
    else:
        command = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.Popen([command, str(path.resolve())], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
