from itertools import product
import math
import torch
import logging
import os
import json
import numpy as np
from typing import List, Dict, Tuple, Any

from config import device_name, config
from utils.utils import ffmpeg_decode_mv_residual, roi_center_to_xyxy
from extractor.extractor import get_inside
from .env_encoder import scheduler_context_size
from .neuralucb import NeuralUCB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handle_bound(annotate, x_lim, y_lim):
    x, y, w, h = annotate
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x + w > x_lim:
        w = x_lim - x
    if y + h > y_lim:
        h = y_lim - y
    return x, y, w, h

def to_tensor(l: List[Any]) -> torch.Tensor:
    return torch.tensor(l, device=device_name, dtype=torch.float32)

def local_residual_motion(roi_xyxy_max, mvs, residual_arr):
    residual_cropped = []
    motion_cropped = []
    def get_ratio_and_next(val, val_lim):
        if val % 16 != 0:
            partial = 16 - val % 16
            ratio = partial / 16
            val += partial
        elif val_lim - val < 16:
            ratio = (val_lim - val) / 16
            val = val_lim
        else:
            ratio = 1
            val += 16
        return ratio, val

    for idx, xyxy in enumerate(roi_xyxy_max):
        residual = 0
        x1, y1, x2, y2 = list(xyxy)
        y = y1
        while y < y2:
            index_y = y // 16
            st_y_ratio, y = get_ratio_and_next(y, y2)
            x = x1
            while x < x2:
                index_x = x // 16
                st_x_ratio, x = get_ratio_and_next(x, x2)
                residual += st_y_ratio * st_x_ratio * float(residual_arr[idx][index_y * 12 + index_x])
        residual_cropped.append(residual / max(config['sr_sizes']) ** 2)
        motion = 0
        for item in mvs[idx]:
            source, w, h, src_x, src_y, dst_x, dst_y, motion_x, motion_y, motion_scale = item
            x_inside = get_inside((dst_x, dst_x + w), (x1, x2))
            y_inside = get_inside((dst_y, dst_y + h), (y1, y2))
            if x_inside > 0 and y_inside > 0:
                motion += 1
        motion_cropped.append(motion)
    return residual_cropped, motion_cropped

def get_residual_info(file: str):
    env_file = file.split('/')
    if os.path.exists("../#semiauto-roi-labeler/180P-annotated/5.9/" + env_file[-2]):
        folder = f"../#semiauto-roi-labeler/180P-annotated/5.9/{env_file[-2]}"
    else:
        folder = f"../#semiauto-roi-labeler/180P-annotated/6.30/{env_file[-2]}"
    with open(f"{folder}/init-stream0.m4s", 'rb') as f:
        init = f.read()
    chunk = f"{folder}/chunk-stream0-0000{env_file[-1]}.m4s"
    if not os.path.exists(chunk):
        return None, None
    with open(chunk, 'rb') as f:
        chunk = f.read()
    with open(folder + ".json", 'r') as f:
        annotation_data = json.load(f)
    ndarray, framerate, mvs, types, residual_arr = ffmpeg_decode_mv_residual(init + chunk, "test")
    # tensors_180p = torch.from_numpy(ndarray.transpose(0, 3, 1, 2))  # NHWC → NCHW
    roi_center = []
    for idx in range(ndarray.shape[0]):
        global_idx = (int(env_file[-1]) - 1) * 90 + idx
        frame_anno = annotation_data['annotations'][str(global_idx)]
        x, y, w, h = handle_bound(frame_anno, 180, 320)
        center_x = x + w // 2
        center_y = y + h // 2
        roi_center.append((center_x, center_y))
    roi_xyxy_max = roi_center_to_xyxy(roi_center, max(config['sr_sizes']), (ndarray.shape[1], ndarray.shape[2]))
    return local_residual_motion(roi_xyxy_max, mvs, residual_arr)


def video_flow(video_tensor: torch.Tensor) -> float:
    with torch.no_grad():
        # 批量转换为灰度图 (N, H, W)
        gray_tensor = 0.299 * video_tensor[:, 0] + 0.587 * video_tensor[:, 1] + 0.114 * video_tensor[:, 2]
        frame_diffs = torch.abs(gray_tensor[1:] - gray_tensor[:-1])
        per_pair_flow = frame_diffs.mean(dim=(1, 2))  # 每对帧的(H,W)空间平均
        overall_flow = per_pair_flow.mean().item()  # 所有帧对的平均
    return overall_flow


class Scheduler:
    def __init__(self, model_path: str = "", neural_ucb=True, net_eval=False,
                 neural_ucb_config: Dict[str, Any] | None = None):
        self.edge_model_num = len(config['edge']['sr_models'])
        self.client_model_num = len(config['client']['sr_models'])
        self.sr_size_list = config['sr_sizes']
        self.decision_space = list(product(list(range(self.edge_model_num + self.client_model_num)), self.sr_size_list))
        self.decision_num = len(self.decision_space)
        self.context_size = scheduler_context_size(self.decision_num)
        self.arm_features = torch.zeros((self.decision_num, 2), device=device_name)

        if neural_ucb:
            self.neural_ucb = NeuralUCB(d=self.context_size, K=self.decision_num, encode_hidden=12, hidden_size=32,
                                        model_path=model_path, **(neural_ucb_config or {}))
        if net_eval:
            self.neural_ucb.net.eval()
        self.cached_contexts = None
        self.cached_action = None

    def encode_one_hot(self, idx, lim):
        one_hot = [0] * lim
        if 0 <= idx < lim:
            one_hot[idx] = 1
        return one_hot

    def __call__(self, env: Dict[str, float | str], sr_time: Dict[Tuple[int, int], float],
                 user_slowest: Dict[int, float]) -> Tuple[int, int]:
        texture_context = to_tensor(
            [env['receive_size'] / 1048576, env['flow'], env['res_mean'], env['res_std'], env['res_diverse'],
             env['mot_mean'] / 100, env['mot_std'] / 100, env['mot_diverse'] / 100, 1 / env['bandwidth']])

        contexts = []
        for idx, decision in enumerate(self.decision_space):
            model, SR_size = decision
            line_ratio = SR_size * math.sqrt(2) / math.sqrt(180 ** 2 + 320 ** 2)
            sr_idx = self.sr_size_list.index(SR_size)
            if model < self.edge_model_num:
                remain = env['buffer'] - sr_time[decision] - env['sr_wait']
                context = torch.cat((texture_context,
                                     to_tensor([line_ratio, env['encode_wait'][sr_idx], remain,
                                                *self.encode_one_hot(idx, self.decision_num)])))
                contexts.append(context)
            else:
                remain = env['buffer'] - user_slowest[SR_size]
                context = torch.cat((texture_context,
                                     to_tensor(
                                         [line_ratio, 0, remain, *self.encode_one_hot(idx, self.decision_num)])))
                contexts.append(context)
        # print(contexts, 111)
        contexts = torch.stack(contexts)
        action = self.neural_ucb.take_action(contexts.to(device_name))
        self.cached_contexts = contexts
        self.cached_action = action

        return self.decision_space[action]

    def update_param(self, reward):
        self.neural_ucb.update(self.cached_contexts, self.cached_action, reward)
