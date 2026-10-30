"""五帧上下文读取；与检测、编码分别维护，内存只保留当前窗口。"""
from collections import deque

import cv2


class FiveFrameReader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        self.cache = deque(maxlen=5)
        self.last_index = -1

    def read(self, target_index):
        if target_index < self.last_index - 2:
            raise ValueError('时序视频必须按帧号递增读取')
        while self.last_index < target_index + 2:
            ok, frame = self.cap.read()
            if not ok:
                break
            self.last_index += 1
            self.cache.append((self.last_index, frame))
        cached = dict(self.cache)
        if target_index not in cached:
            raise ValueError(f'时序窗口缺少目标帧: {target_index}')
        # 与作者数据集一致：越界的支持帧以中心帧替代。
        indices = [i if i in cached else target_index for i in range(target_index - 2, target_index + 3)]
        return [cached[i] for i in indices], indices, max(indices)

    def close(self):
        self.cap.release()
