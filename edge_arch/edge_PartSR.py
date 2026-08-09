import math
import time

import numpy as np
import torch
import struct

from torchvision import transforms
from torch import multiprocessing as mp
from typing import Dict, Any, Tuple

from super_resolution.infer import generate_sr_patch, Inferrer
from utils.mp4_bin_editor import update_reencode_metadata
from utils.utils import ffmpeg_decode_mv_residual, ffmpeg_tensor_to_bytes, roi_center_to_xyxy
from classifier.classifier import Classifier
from extractor.extractor import RoIExtractor
from extractor.detector import Detector
from scheduler.formal_config import formal_neural_ucb_config
from scheduler.scheduler import Scheduler, local_residual_motion, video_flow
from config import config
from .edge import Edge
from .sr_dispatcher import Task, run_dispatcher

def crop_rois(video_tensor, bboxes):
    device = video_tensor.device
    N, C, H, W = video_tensor.shape

    # 确保bboxes是整数类型
    if bboxes.dtype != torch.long:
        bboxes = bboxes.long()

    # 计算裁剪尺寸（假设所有ROI尺寸相同）
    crop_h = int(bboxes[0, 3] - bboxes[0, 1])
    crop_w = int(bboxes[0, 2] - bboxes[0, 0])

    # 创建索引网格
    ii = torch.arange(N, device=device)[:, None, None, None]  # (N, 1, 1, 1)
    jj = torch.arange(C, device=device)[None, :, None, None]  # (1, C, 1, 1)

    # 高度索引 [N, 1, h, 1]
    h_idx = bboxes[:, 1].view(N, 1, 1, 1) + torch.arange(crop_h, device=device).view(1, 1, crop_h, 1)

    # 宽度索引 [N, 1, 1, w]
    w_idx = bboxes[:, 0].view(N, 1, 1, 1) + torch.arange(crop_w, device=device).view(1, 1, 1, crop_w)

    # 执行向量化裁剪（自动广播到[N, C, h, w]）
    cropped = video_tensor[ii, jj, h_idx, w_idx]

    return cropped

# 并行的超分辨率处理函数
def mp_server_sr(sr_queue, pipe_conn, task_queue):
    torch.manual_seed(42)
    inferrer = Inferrer("edge")
    # 将 run_benchmark 的结果发送回主进程
    pipe_conn.send(inferrer.run_benchmark())
    pipe_conn.close()  # 关闭管道连接
    while True:
        placeholder = sr_queue.get()
        if placeholder is None:
            break
        data = placeholder.get_data()
        if data is None:
            continue
        tmp_pipe, tensors, action = data
        task_queue.put((Task.START, placeholder.number))
        bg = time.perf_counter()
        try:
            result = inferrer(tensors, action=action)
            tmp_pipe.send((result, time.perf_counter() - bg))
        except Exception as exc:
            try:
                tmp_pipe.send((None, time.perf_counter() - bg, f"{type(exc).__name__}: {exc}"))
            except (BrokenPipeError, EOFError, OSError):
                pass
        finally:
            tmp_pipe.close()
            task_queue.put((Task.FINISH, placeholder.number))


class EdgePartSR(Edge):
    @staticmethod
    def static_init():
        mp.set_start_method('spawn', force=True)
        torch.multiprocessing.set_sharing_strategy('file_descriptor')
        parent = Edge.static_init()

        initial_encode_seconds = {60: 0.25, 90: 0.45}
        try:
            encode_samples = [initial_encode_seconds[size] for size in config["sr_sizes"]]
        except KeyError as exc:
            raise ValueError(f"Missing initial encode time for SR size {exc.args[0]}") from exc
        hist_encode = mp.Array('f', encode_samples)

        # create Pipe & SR subprocess, and get benchmark result
        task_queue = mp.Queue()
        parent_benchmark, child_benchmark = mp.Pipe()
        proc_dispatcher = mp.Process(target=run_dispatcher, args=(task_queue, child_benchmark, mp_server_sr))
        proc_dispatcher.start()
        child_benchmark.close()
        try:
            edge_sr_time = parent_benchmark.recv()
        except BaseException:
            task_queue.put((None, None))
            proc_dispatcher.join(timeout=10)
            if proc_dispatcher.is_alive():
                proc_dispatcher.terminate()
                proc_dispatcher.join(timeout=5)
            raise
        finally:
            parent_benchmark.close()
        schedule_lock = mp.Lock()

        return (hist_encode, task_queue, proc_dispatcher, edge_sr_time, schedule_lock), parent


    def __init__(self, static_values):
        self_static, parent_static = static_values
        super().__init__(parent_static)

        self.sr_model_list = config["edge"]["sr_models"]
        self.sr_sizes = config["sr_sizes"]
        self.sr_model_number = len(self.sr_model_list)

        self.classifier = Classifier(config["classify_model_path"], num_classes=3)
        self.detector = Detector(config["detector_model_path"])
        self.scheduler = Scheduler(
            config["scheduler_model_path"],
            net_eval=True,
            neural_ucb_config=formal_neural_ucb_config(),
        )

        hist_encode, task_queue, proc_dispatcher, edge_sr_time, schedule_lock = self_static
        self.hist_encode = hist_encode
        self.task_queue = task_queue
        self.proc_dispatcher = proc_dispatcher
        self.edge_sr_time = edge_sr_time
        self.schedule_lock = schedule_lock

    def close(self):
        process = getattr(self, 'proc_dispatcher', None)
        if process is None or not process.is_alive():
            return
        self.task_queue.put((None, None))
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    def init_args(self, identifier: str, args: Dict[str, Any]):
        user_id = args['user_id']
        client_sr_time = self.manager.dict()
        for k, v in args['SR_time'].items():
            k_int = int(k)
            assert k_int in config["sr_sizes"], "sr_sizes at client doesn't match that at edge"
            client_sr_time[k_int] = v
        with self.streamer_lock:
            self.streamers[identifier][user_id] = client_sr_time

    def __get_slowest(self, identifier: str) -> Dict[int, float]:
        client_sr_time = {k: -math.inf for k in config["sr_sizes"]}
        with self.streamer_lock:
            sr_time_data = self.streamers[identifier].items()
        for key, value in sr_time_data:
            if not isinstance(key, str) or len(key) != 36:
                continue
            for size, cost in value.items():
                # Client SR time is a latency in seconds.  Use the worst-case
                # client so the scheduler does not overestimate client slack.
                client_sr_time[size] = max(client_sr_time[size], cost)
        return client_sr_time

    def __update_encode_time(self, new_time, sr_idx):
        self.hist_encode.acquire()
        old_time = self.hist_encode[sr_idx]
        self.hist_encode[sr_idx] = (old_time + new_time) / 2
        self.hist_encode.release()

    def _handle(self, identifier: str, received: bytes, args: Dict) -> bytes:
        buffer_timer = time.perf_counter()
        with self.streamer_lock:
            if identifier not in self.streamers:
                raise RuntimeError("Found no header " + identifier)
            print(f'[PartSR] received {len(received)} bytes')
            status = self.streamers[identifier]
        video_bytes = status["header"] + received
        ffmpeg_identifier = f"{identifier}_{args['user_id']}"

        # 1. decode to tensor & extract MV, RE
        bg = time.perf_counter()
        ndarray, framerate, mvs, types, residual_arr = ffmpeg_decode_mv_residual(video_bytes, ffmpeg_identifier)
        tensors_div_255 = torch.from_numpy(ndarray.transpose(0, 3, 1, 2)).float().div(255)  # NHWC → NCHW
        tensors_normalized = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
            tensors_div_255)
        t_decode = time.perf_counter() - bg
        print(f"[PartSR] 1.decode: {t_decode:.3f}s, shape: {ndarray.shape}")

        # 2. classify the stream if it hasn't been classified
        t_classify = 0
        if "class" not in status:
            bg = time.perf_counter()
            classified = self.classifier(tensors_normalized)
            with self.streamer_lock:
                self.streamers[identifier]["class"] = classified
            t_classify = time.perf_counter() - bg
            print(f"[PartSR] 2.classify: {t_classify:.3f}s, classified: {classified}")

        # 3. extract RoI
        bg = time.perf_counter()
        extractor = RoIExtractor(tensors_normalized, ndarray, types, mvs, framerate, residual_arr, self.detector)
        roi_array = extractor.run()
        roi_xyxy_max = roi_center_to_xyxy(roi_array, max(self.sr_sizes), (ndarray.shape[1], ndarray.shape[2]))
        t_extract = time.perf_counter() - bg
        print(f"[PartSR] 3.extract: {t_extract:.3f}s")

        # 4, prepare video features for scheduling
        bg = time.perf_counter()
        biggest_cropped = crop_rois(tensors_div_255, torch.from_numpy(roi_xyxy_max))
        v_flow = video_flow(biggest_cropped)
        residual_cropped, motion_cropped = local_residual_motion(roi_xyxy_max, mvs, residual_arr)
        t_prep = time.perf_counter() - bg
        print(f"[PartSR] 4.schedule preparation: {t_prep:.6f}s")

        # 5. choose proper SR action according to environment and video features
        bg = time.perf_counter()
        env = {
            'receive_size': len(received),
            'flow': v_flow,

            'res_mean': np.mean(residual_cropped),
            'res_std': np.std(residual_cropped),
            'res_diverse': np.mean(np.abs(np.diff(residual_cropped))),
            'mot_mean': np.mean(motion_cropped),
            'mot_std': np.std(motion_cropped),
            'mot_diverse': np.mean(np.abs(np.diff(motion_cropped))),

            'bandwidth': args['bandwidth'] * 0.8,
            'encode_wait': [x * 1.2 for x in list(self.hist_encode)],
        }
        parent_task, child_task = mp.Pipe()
        with self.schedule_lock:  # assure waiting time for schedule = waiting time when SR
            self.task_queue.put((Task.ACQUIRE, child_task))
            t_wait = parent_task.recv()
            parent_task.close()
            dynamic_env = {
                'buffer': max(args['buffer'] - (time.perf_counter() - buffer_timer), 0),
                'sr_wait': t_wait,
            }
            env.update(dynamic_env)
            # print(env)
            action, SR_size = self.scheduler(env, self.edge_sr_time, self.__get_slowest(identifier))
            if action < self.sr_model_number:
                parent_reserve, child_reserve = mp.Pipe()
                self.task_queue.put((Task.CREATE, (child_reserve, self.edge_sr_time[(action, SR_size)])))
                number = parent_reserve.recv()
                parent_reserve.close()
        if SR_size == max(self.sr_sizes):
            roi_xyxy = roi_xyxy_max
        else:
            roi_xyxy = roi_center_to_xyxy(roi_array, SR_size, (ndarray.shape[1], ndarray.shape[2]))
        t_schedule = time.perf_counter() - bg
        print(f"[PartSR] 5.schedule: {t_schedule:.3f}s (wait {t_wait}s), action: {action}, SR_size: {SR_size}")

        if action < self.sr_model_number:  # 5. if SR at the server
            bg = time.perf_counter()
            sent_to_worker = False
            sr_parent = None
            try:
                tensor_for_SR = generate_sr_patch(tensors_div_255, roi_xyxy)
                sr_parent, sr_child = mp.Pipe()
                self.task_queue.put((Task.SEND, (number, (sr_child, tensor_for_SR, action))))
                sent_to_worker = True
                worker_response = sr_parent.recv()
                if len(worker_response) == 3 and worker_response[0] is None:
                    raise RuntimeError(f"edge SR worker failed: {worker_response[2]}")
                sr_tensor, elapsed = worker_response
            except BaseException:
                if not sent_to_worker:
                    self.task_queue.put((Task.CANCEL, number))
                raise
            finally:
                if sr_parent is not None:
                    sr_parent.close()
            self.edge_sr_time[(action, SR_size)] = (elapsed + self.edge_sr_time[(action, SR_size)]) / 2  # update
            t_sr = time.perf_counter() - bg
            print(f"[PartSR] 6.super resolution: {t_sr:.3f}s, tensor: {sr_tensor.shape}")

            bg = time.perf_counter()
            reencode = ffmpeg_tensor_to_bytes(sr_tensor, framerate, ffmpeg_identifier)
            final_video = update_reencode_metadata(video_bytes, reencode)
            t_encode = time.perf_counter() - bg
            self.__update_encode_time(t_encode, self.sr_sizes.index(SR_size))
            print(f"[PartSR] 7.re-encode {t_encode:.3f}s, encoded patch size: {len(final_video)}")

            response_data = struct.pack('>BII', int(0), SR_size, len(final_video)) + final_video
        else:  # 5. if SR at the client
            response_data = struct.pack('>BI', int(1), SR_size)

        for i in range(ndarray.shape[0]):
            response_data += struct.pack('>II', roi_xyxy[i][0], roi_xyxy[i][1])
        return struct.pack('>I', len(received)) + received + response_data
