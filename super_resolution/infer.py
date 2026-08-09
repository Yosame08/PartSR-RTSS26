import math

from .span_arch import SPAN
from basicsr.archs.edsr_arch import EDSR
from basicsr.archs.basicvsr_arch import BasicVSR

from typing import Tuple, List, Dict
from collections import OrderedDict

import os
import tqdm
import torch
import time
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import device_name, config
device = torch.device(device_name)
pad = config["pad"]
prefixes = ["super_resolution/models/", "models/"]

def load_basicsr_model(checkpoint):
    preference = ["params", "params_ema"]
    if preference[0] in checkpoint:  # BasicSR 保存的权重文件
        state_dict = checkpoint[preference[0]]
    elif preference[1] in checkpoint:
        state_dict = checkpoint[preference[1]]
    else:
        state_dict = checkpoint

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def generate_sr_patch(tensors: torch.Tensor, roi_list: np.ndarray) -> torch.Tensor:
    """
    CPU生成用于超分的图像批次

    Args:
        tensors: 输入图像批次 (B, C, H, W)
        roi_list: 每张图像的ROI坐标 [(x1, y1, x2, y2), ...]

    Returns:
        (B, C, H + pad * 2, W + pad * 2)
    """
    B, C, H, W = tensors.shape
    SR_size = roi_list[0][2] - roi_list[0][0]  # 计算 ROI 的宽度
    pad_size = SR_size + pad * 2

    # 阶段2: 裁剪 ROI 区域
    lr_patches = []
    for batch_idx in range(B):
        x1, y1, x2, y2 = roi_list[batch_idx]

        x1_exp = max(0, x1 - pad)
        y1_exp = max(0, y1 - pad)
        x2_exp = min(W, x2 + pad)
        y2_exp = min(H, y2 + pad)
        patch = tensors[batch_idx, :, y1_exp:y2_exp, x1_exp:x2_exp]  # (C, H, W)
        pad_left = pad - (x1 - x1_exp)
        pad_right = pad - (x2_exp - x2)
        pad_top = pad - (y1 - y1_exp)
        pad_bottom = pad - (y2_exp - y2)
        padded = torch.nn.functional.pad(patch, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')

        lr_patches.append(padded)
        assert (padded.shape[-2], padded.shape[-1]) == (pad_size, pad_size), f"SR input shape error: {(x1, y1, x2, y2)}, shape: {padded.shape}, pad={pad}"

    # 阶段3: 批量超分推理
    return torch.stack(lr_patches, dim=0)


class Inferrer:
    def __init__(self, device_type: str, pad_size: int=pad, pad_crop: bool=True):
        assert device_type == 'edge' or device_type == 'client', 'device must be edge or client'
        self.pad_size = pad_size
        self.pad_crop = pad_crop
        model_names = config[device_type]['sr_models']
        self.models: List[torch.nn.Module | None] = []
        self.vsr_models = []
        self.batch_size = config[device_type]['batch_size']
        for name in model_names:
            chosen = ""
            for prefix in prefixes:
                path = os.path.join(prefix, name)
                print(path)
                if os.path.exists(path):
                    chosen = path
                    break
            name = name.lower()
            if name.startswith("edsr"):
                if 's' in name[4:]:
                    model = EDSR(3, 3, 32, 8)
                elif 'm' in name[4:]:
                    model = EDSR(3, 3, 64, 16)
                elif 'l' in name[4:]:
                    model = EDSR(3, 3, 256, 32, res_scale=0.1)
                else:
                    raise NotImplementedError(f"Cannot resolve EDSR type (M/L): {name}")
            elif name.startswith("span"):
                if "ch48" in name:
                    feature_channels = 48
                elif "ch52" in name:
                    feature_channels = 52
                else:
                    raise NotImplementedError(f"Cannot resolve span channels (ch48/ch52): {name}")
                if "x2" in name:
                    scale = 2
                elif "x4" in name:
                    scale = 4
                else:
                    raise NotImplementedError(f"Cannot resolve span scale (x2/x4): {name}")
                model = SPAN(3, 3, feature_channels, scale)
            elif name.startswith("basicvsr"):
                model = BasicVSR(64, 30, "super_resolution/spynet_sintel_final-3d2a1287.pth")
            else:
                raise NotImplementedError(f"Cannot resolve model type: {name}")

            if chosen == "":
                self.models.append(None)
                self.vsr_models.append(None)
                logger.warning(f"Cannot find model file: {name}")
            else:
                state_dict = torch.load(chosen, map_location="cpu", weights_only=True)
                print(model.load_state_dict(load_basicsr_model(state_dict), strict=True))
                model = model.to(device)
                for param in model.parameters():
                    param.requires_grad = False
                self.models.append(model)
                self.vsr_models.append(isinstance(model, BasicVSR))

    def run_benchmark(self) -> Dict[Tuple[int, int], float]:
        print("running benchmark...")
        sizes = config['sr_sizes']
        ret = {}
        for idx, model in enumerate(self.models):
            time.sleep(0.01)
            for h in tqdm.tqdm(sizes):
                with torch.no_grad():
                    dummy_input = torch.randn(config['chunk_len'] * 30, 3, h + pad * 2, h + pad * 2).to(device)
                    __ = self(dummy_input, idx) # warmup
                    result = []
                    for _ in range(6):
                        start_time = time.perf_counter()
                        __ = self(dummy_input, idx)
                        result.append(time.perf_counter() - start_time)
                    result_sorted = sorted(result)
                    trimmed = result_sorted[1:-1]  # exclude minimum and maximum
                    mean = sum(trimmed) / len(trimmed)
                    variance = sum((x - mean) ** 2 for x in trimmed) / len(trimmed)
                    print(f"Model {idx} output shape: {__.shape}, mean={mean}s, std. dev.={math.sqrt(variance)}")
                    ret[(idx, h)] = mean
        print("[PartSR SR process] benchmark done")
        return ret

    def __call__(self, tensors: torch.Tensor, action: int, scale_factor: int = 4) -> torch.Tensor:
        """
        Args:
            tensors: input patches (B, C, H, W)
            action: choice of SR model

        Returns:
            HR patches (B, C, H*4, W*4)
        """
        b = self.batch_size[action]
        pos_low = self.pad_size * scale_factor
        pos_high = (tensors.shape[-1] - self.pad_size) * scale_factor
        res = []
        with torch.no_grad():
            if self.vsr_models[action]:
                for i in range(0, tensors.size(0), b):
                    batch = tensors[i:i + b]
                    sr_part = self.models[action](batch.unsqueeze(0).to(device))
                    if self.pad_crop:
                        sr_part = sr_part[0, :, :, pos_low:pos_high, pos_low:pos_high]
                    res.append(sr_part)
            else:
                for i in range(0, tensors.size(0), b):
                    batch = tensors[i:i + b]
                    sr_part = self.models[action](batch.to(device))
                    if self.pad_crop:
                        sr_part = sr_part[:, :, pos_low:pos_high, pos_low:pos_high]
                    res.append(sr_part)

        sr_patches = torch.cat(res, dim=0)  # (B, C, H*4, W*4)

        return sr_patches.cpu()
