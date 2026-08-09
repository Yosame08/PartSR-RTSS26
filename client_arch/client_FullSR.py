import torch
import numpy as np
import time
import multiprocessing as mp
from basicsr.archs.edsr_arch import EDSR
from utils.utils import decord_bytes2numpy
from utils.sr_output import sr_tensor_to_uint8_rgb
from config import config, device_name
from .client import Client
import gc

sr_model_path = config['client']['sr_models'][0]

def process_sr(sr_queue):
    if not sr_model_path.lower().startswith('edsr'):
        raise RuntimeError('only EDSR model is supported')
    if 's' in sr_model_path[4:].lower():
        model = EDSR(3, 3, 32, 8)
    elif 'm' in sr_model_path[4:].lower():
        model = EDSR(3, 3, 64, 16)
    elif 'l' in sr_model_path[4:].lower():
        model = EDSR(3, 3, 256, 32, res_scale=0.1)
    else:
        raise NotImplementedError(f"Cannot resolve EDSR type (M/L): {sr_model_path}")
    state_dict = torch.load("super_resolution/models/" + sr_model_path, map_location="cpu", weights_only=True)
    print(model.load_state_dict(state_dict['params'], strict=True))
    model = model.to(device_name)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    while True:
        tmp_pipe, tensors = sr_queue.get()
        if tmp_pipe is None:
            break
        quad = tensors.shape[0] // 4
        tensors = tensors.to(device_name)
        with torch.no_grad():
            sr_tensors1 = model(tensors[:quad]).cpu()
            sr_tensors2 = model(tensors[quad:quad*2]).cpu()
            sr_tensors3 = model(tensors[quad*2:quad*3]).cpu()
            sr_tensors4 = model(tensors[quad*3:]).cpu()
            sr_tensors = torch.cat([sr_tensors1, sr_tensors2, sr_tensors3, sr_tensors4], dim=0)
        tmp_pipe.send(sr_tensors)
        tmp_pipe.close()
        gc.collect()

class ClientFullSR(Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mp.set_start_method('spawn', force=True)
        torch.multiprocessing.set_sharing_strategy('file_descriptor')
        self.sr_queue = mp.Queue()
        try:
            process_b = mp.Process(target=process_sr, args=(self.sr_queue,))
            process_b.start()
            time.sleep(6) # wait for boot
        except Exception as e:
            print(f"An error occurred when creating EdgePartSR: {e}")
            self.sr_queue.put((None, None))
            process_b.terminate()
            exit(-1)

    def _handle(self, received: bytes) -> np.ndarray:
        numpy_180p, fps_ = decord_bytes2numpy(self.init_chunk + received)
        tensor_for_SR = torch.from_numpy(numpy_180p.astype(np.float32)) / 255
        sr_parent_pipe, sr_child_pipe = mp.Pipe()
        self.sr_queue.put((sr_child_pipe, tensor_for_SR))
        sr_tensor = sr_parent_pipe.recv()
        sr_parent_pipe.close()
        numpy_720p = sr_tensor_to_uint8_rgb(sr_tensor)
        return numpy_720p

    def __del__(self):
        self.sr_queue.put((None, None))
