import math

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

from utils.utils import decord_bytes2numpy
from config import config

class Client:
    def __init__(self, init_buffer: int, *args, **kwargs):
        self.cached = []
        self.played = 0
        self.buffer = init_buffer
        self.init_chunk = None # only video stream

    def receive(self, received: bytes, is_init: bool):
        if is_init:
            self.init_chunk = received
        else:
            handled = self._handle(received)
            self.cached.append(handled)

    def save_video_rebuffer(self, lag_info: List[float], filename: str, annotate: bool, gt_roi: List[Tuple]=None):
        if annotate:
            assert gt_roi is not None, "must set gt_roi to annotate RoI"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = self.cached[0].shape[3]
        height = self.cached[0].shape[2]
        video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))

        frame = np.zeros_like(self.cached[0][0].transpose(1, 2, 0))
        for idx, clip in enumerate(self.cached):
            if lag_info[idx] > 0:
                # 计算卡顿帧数（向上取整）
                lag_frames = math.ceil(lag_info[idx] * 30)
                last_frame = frame.copy()  # 获取当前块的最后一帧
                print(f"lag for {lag_frames} frames")
                # 在卡顿期间重复最后一帧并添加加载动画
                for j in range(lag_frames):
                    lag_frame = last_frame.copy()
                    video_writer.write(lag_frame)

            for i in range(clip.shape[0]):
                global_idx = clip.shape[0] * idx + i
                # 转换为OpenCV格式 (HWC, BGR)
                assert self.cached[idx][i].shape == (3, 1280, 720), self.cached[idx][i].shape
                frame = self.cached[idx][i].transpose(1, 2, 0)[:, :, ::-1].copy()  # RGB to BGR
                frame = np.ascontiguousarray(frame, dtype=np.uint8)
                if annotate:
                    x, y, w, h = gt_roi[global_idx]
                    x1_gt = int((x - w / 2) * width)
                    y1_gt = int((y - h / 2) * height)
                    x2_gt = x1_gt + int(w * width)
                    y2_gt = y1_gt + int(h * height)
                    cv2.rectangle(frame, (x1_gt, y1_gt), (x2_gt, y2_gt), (0, 0, 255), 2)  # 绘制真实ROI (红色框)
                    cv2.putText(frame, "Ground Truth ROI", (x1_gt, y1_gt - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                # cv2.putText(frame, f"Frame: {i}", (10, 30),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                video_writer.write(frame)
        video_writer.release()

    def _handle(self, received: bytes) -> np.ndarray:
        """
        If the architecture needs to do something at the client, override this function.
        Process the chunk and return a numpy array (NCHW, RGB). The array will be automatically appended to cache.

        If this function is not overridden, it will concatenate the init_chunk and whatever it receives from the server,
        and try to decode the bytes.

        :param received: Binary data from server
        """
        video_bytes = received
        video_np, fps = decord_bytes2numpy(video_bytes, threads=1)
        print(video_np.shape)
        n, c, h, w  = video_np.shape
        if h == config['video_height'] * 4 and w == config['video_width'] * 4:
            return video_np
        else:
            print(f"client.py: received video doesn't match 720p, perform x4 scaling")
            target_h, target_w = config['video_height'] * 4, config['video_width'] * 4
            video_np = video_np.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            resized_frames = []
            for frame in video_np:
                # 使用双三次插值缩放
                resized_frame = cv2.resize(
                    frame,
                    (target_w, target_h),
                    interpolation=cv2.INTER_LINEAR
                )
                resized_frames.append(resized_frame)
            return np.array(resized_frames).transpose(0, 3, 1, 2)


    def init_arg(self, **kwargs) -> Dict[str, Any]:
        """
        If the architecture needs to send some arguments to the edge at the beginning, override this function.
        """
        return {}

    def common_arg(self, **kwargs) -> Dict[str, Any]:
        """
        If the architecture needs to send some arguments to the edge every time it sends a request, override this function.
        """
        return {}