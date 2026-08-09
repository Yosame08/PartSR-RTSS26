import torch
import multiprocessing as mp
import queue
import time

from basicsr.archs.edsr_arch import EDSR

from utils.mp4_bin_editor import update_reencode_metadata
from utils.utils import decord_bytes2numpy, ffmpeg_tensor_to_bytes
from .edge import Edge
from config import config, device_name

sr_model_path = config['edge']['sr_models'][0]


def process_sr(sr_queue, status_queue):
    try:
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
    except Exception as exc:
        status_queue.put((False, repr(exc)))
        raise

    status_queue.put((True, device_name))
    while True:
        tmp_pipe, tensors = sr_queue.get()
        if tmp_pipe is None:
            break

        with torch.no_grad():
            sr_tensors = model(tensors.to(device_name)).cpu()
        tmp_pipe.send(sr_tensors)
        tmp_pipe.close()


def validate_lr_frames(frames):
    expected = (3, config['video_height'], config['video_width'])
    if frames.ndim != 4 or tuple(frames.shape[1:]) != expected:
        raise ValueError(
            f"EdgeFullSR expected NCHW low-resolution frames with spatial shape {expected}, "
            f"got {frames.shape}"
        )


class EdgeFullSR(Edge):
    worker_start_timeout = 60
    worker_result_timeout = 300

    def __init__(self, static_values):
        mp.set_start_method('spawn', force=True)
        torch.multiprocessing.set_sharing_strategy('file_descriptor')
        super().__init__(static_values)
        self.sr_queue = mp.Queue()
        status_queue = mp.Queue()
        self.sr_process = mp.Process(target=process_sr, args=(self.sr_queue, status_queue))
        self.sr_process.start()
        try:
            ready, detail = status_queue.get(timeout=self.worker_start_timeout)
        except queue.Empty as exc:
            self.close()
            raise RuntimeError("EdgeFullSR worker did not start within 60 seconds") from exc
        if not ready:
            self.close()
            raise RuntimeError(f"EdgeFullSR worker failed to start: {detail}")
        print(f"EdgeFullSR worker ready on {detail}")

    def _handle(self, identifier: str, received: bytes, args) -> bytes:
        if identifier not in self.streamers:
            raise RuntimeError("Found no header " + identifier)
        status = self.streamers[identifier]
        video_bytes = status["header"] + received

        lr_numpy, fps = decord_bytes2numpy(video_bytes)
        validate_lr_frames(lr_numpy)
        lr_tensor = torch.from_numpy(lr_numpy).float() / 255
        sr_parent_pipe, sr_child_pipe = mp.Pipe()
        self.sr_queue.put((sr_child_pipe, lr_tensor))
        deadline = time.monotonic() + self.worker_result_timeout
        while not sr_parent_pipe.poll(0.5):
            if not self.sr_process.is_alive():
                sr_parent_pipe.close()
                raise RuntimeError("EdgeFullSR worker exited during inference")
            if time.monotonic() >= deadline:
                sr_parent_pipe.close()
                raise TimeoutError("EdgeFullSR inference exceeded 300 seconds")
        sr_tensor = sr_parent_pipe.recv()
        sr_parent_pipe.close()

        reencode = ffmpeg_tensor_to_bytes(sr_tensor.clamp_(0, 1), fps, identifier)
        final_video = update_reencode_metadata(video_bytes, reencode)

        return final_video

    def close(self):
        process = getattr(self, 'sr_process', None)
        if process is None or not process.is_alive():
            return
        self.sr_queue.put((None, None))
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
