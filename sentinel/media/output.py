"""单个素材的标框保存与预览；每次只打开一个视频编码器。"""
import tempfile
from pathlib import Path

import cv2

from .video import MP4Writer, open_video


class ResultOutput:
    def __init__(self, out, name, path, *, save, show, fps=None, caption=None):
        self.save, self.show, self.fps = save, show, fps
        self.caption = caption
        self.writer = None
        self.path = out / f'{name}_{path.name}.mp4' if fps else out / f'{name}_{path.name}.jpg'
        if fps and show and not save:
            self.path = Path(tempfile.mkdtemp(prefix='chicken_preview_')) / self.path.name

    def __enter__(self):
        return self

    def write(self, result):
        if not (self.save or self.show):
            return
        plotted = result.plot()
        if self.caption:
            cv2.putText(plotted, self.caption, (12, max(24, plotted.shape[0] - 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2, cv2.LINE_AA)
        if self.fps is not None:
            if self.writer is None:
                height, width = plotted.shape[:2]
                self.writer = MP4Writer(self.path, self.fps, (width, height))
            self.writer.write(plotted)
        else:
            if self.save and len(result.boxes):
                if not cv2.imwrite(str(self.path), plotted):
                    raise OSError(f'保存图片失败: {self.path}')
            if self.show:
                title = self.path.name
                height, width = plotted.shape[:2]
                scale = min(1., 1280 / width, 800 / height)
                cv2.namedWindow(title, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(title, int(width * scale), int(height * scale))
                cv2.imshow(title, plotted)
                cv2.waitKey(0)

    def __exit__(self, exc_type, exc, tb):
        if self.writer is not None:
            self.writer.release()
        if self.show:
            cv2.destroyAllWindows()
            if self.writer is not None and exc_type is None:
                open_video(self.path)
