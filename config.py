import os
import yaml
import torch
from typing import Dict

def parse_device_name(conf: Dict) -> str:
    override = os.environ.get("PARTSR_GPU_ID")
    if override is None:
        gpu_id = conf["gpu_id"]
    else:
        try:
            gpu_id = int(override)
        except ValueError as exc:
            raise ValueError("PARTSR_GPU_ID must be an integer") from exc
    device_name = f"cuda:{gpu_id}" if (torch.cuda.is_available() and 0 <= gpu_id < torch.cuda.device_count()) else "cpu"
    print(f"Specified device: {device_name}")
    return device_name

config_path = "config.yaml" if os.path.exists("config.yaml") else "../config.yaml"
with open(config_path, 'r') as cf:
    config = yaml.load(cf, Loader=yaml.FullLoader)
device_name = parse_device_name(config)
