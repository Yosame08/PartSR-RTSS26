import os
# Keep both GPUs visible for the historical two-process setup, while allowing
# callers to provide a container- or scheduler-specific device mask.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')

from concurrent.futures import ThreadPoolExecutor
import math
import re
import requests
import struct
import time
import csv
import numpy as np

from typing import Dict, Tuple, List, Any, Union
from skimage.metrics import structural_similarity as ssim

# 先导入config和torch
from config import config
import torch

# 然后导入其他模块
from utils.utils import decord_video2numpy
from client_arch.client_PartSR import ClientPartSR

print("All imports successful!")

client_arch = ClientPartSR

datasets = {
    'sell': ['plant', 'beef', 'beef-jerky', 'blueberry', 'drone', 'durian', 'garlic', 'multigoods', 'slipper', 'toycar']
}

# Shared testbed parameters
test_len = 24
chunk_len = config['chunk_len']
buffer_lim = 5
ssim_calc_thread = 16

mu1 = 0.5  # RoI ssim weight
mu2 = 0.5 # Global ssim weight
# Keep the historical scalar used by scheduler-training reward shaping.
mu = 1.0
lambda1 = 3.5  # ssim difference coefficient
lambda2 = 0.4  # rebuffer coefficient

# 网络trace数据
network_stat = [11256, 13946, 12408, 13777, 11297, 7744, 13324, 4796, 7356, 11768, 13922, 12629, 11842, 16473, 14448, 18178, 16397, 15147, 11007, 9258, 9854, 6110, 11633, 12596, 6258]
bandwidth = network_stat[0] / 1024 / 8
net_time = 0

# 修复网络时延计算
def get_time(size):
    global net_time
    ans = 0
    while size > 0:
        speed = network_stat[net_time % len(network_stat)] * 128
        net_time += 1
        if speed >= size:
            return ans + size / speed
        ans += 1
        size -= speed
    return ans

def calc_buffer(buffer, delay, sim_sleep=True, reduce=0.1):
    """Advance the playback buffer using one chunk's end-to-end delay."""
    buffer -= delay
    if buffer < 0:
        rebuffer = -buffer
        buffer = chunk_len
    else:
        rebuffer = 0
        buffer += chunk_len
    if buffer > buffer_lim:
        if sim_sleep:
            time.sleep(max(buffer - buffer_lim - reduce, 0))
        buffer = buffer_lim
    return buffer, rebuffer

def gen_file_info(dataset: str, name: str):
    data_root = os.environ.get('PARTSR_DATA_ROOT', 'server_folder')
    local_folder = os.path.join(data_root, f'{dataset}_{chunk_len}s', name)

    # 修改视频保存路径到指定的dataset文件夹
    video_save_dir = f'client_result/{dataset}/{name}'
    os.makedirs(video_save_dir, exist_ok=True)

    return {
        'local_folder': local_folder,
        'local_gt': os.path.join(local_folder, f'720p-{dataset}-{name}.mp4'),
        'local_lr': os.path.join(local_folder, '180P.mp4'),
        'roi_annotation': os.path.join(local_folder, f'{name}_annotate/labels'),
        'server': f'{config["edge_server"]}/{dataset}_{chunk_len}s/{name}/',
        'video_save': f'{video_save_dir}/ours-{dataset}-{name}.mp4',
        'video_save_dir': video_save_dir,  # 添加目录路径
    }


def find_max_chunk_number(folder_path: str) -> int:
    """Return the largest numbered DASH media chunk in ``folder_path``."""
    pattern = re.compile(r"chunk-stream\d+-(\d+)\.m4s")
    max_number = -1
    for filename in os.listdir(folder_path):
        match = pattern.fullmatch(filename)
        if match is not None:
            max_number = max(max_number, int(match.group(1)))
    return max_number

def calc_qoe(ssim_metric: List[List[Union[int, float]]], rebuffer: List[float], video_duration: float):
    """
    计算QoE指标
    Args:
        ssim_metric: SSIM指标列表，每个元素为[idx, global_ssim, roi_ssim]
        rebuffer: 卡顿时间列表
        video_duration: 原始视频总时长（秒）
    """

    def weighted_ssim(line: List[Union[int, float]]):
        # 使用新的权重公式：mu1*global_ssim + mu2*roi_ssim
        return mu1 * line[1] + mu2 * line[2]

    last_weighted = weighted_ssim(ssim_metric[0])
    ssim_avg = last_weighted
    ssim_dif_avg = 0
    rebuffer_sum = sum(rebuffer)

    for i in range(1, len(ssim_metric)):
        weighted = weighted_ssim(ssim_metric[i])
        ssim_avg += weighted
        ssim_dif_avg += math.fabs(weighted - last_weighted)
        last_weighted = weighted

    # 平均SSIM
    ssim_avg = ssim_avg / len(ssim_metric)
    ssim_dif_avg = ssim_dif_avg / len(ssim_metric)

    # rebuffer除以视频总时长而不是总帧数
    rebuffer_normalized = rebuffer_sum / video_duration

    # 计算QoE
    qoe = ssim_avg - lambda1 * ssim_dif_avg - lambda2 * rebuffer_normalized

    return qoe

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    return ssim(img1, img2, channel_axis=0, data_range=255)

def calc_ssim_worker(args: Tuple[int, np.ndarray, np.ndarray, List[Tuple]]) -> List[Tuple[int, float, float]]:
    start_idx, gt_chunk, processed_chunk, roi_chunk = args
    chunk_metrics = []
    for i in range(gt_chunk.shape[0]):
        idx = start_idx + i
        if i < len(roi_chunk):
            x, y, w, h = roi_chunk[i]
        else:
            x, y, w, h = 0.5, 0.5, 0.3, 0.3

        H, W = gt_chunk.shape[2], gt_chunk.shape[3]  # NCHW format

        # 计算 ROI 边界
        x1 = max(0, int((x - w/2) * W))
        y1 = max(0, int((y - h/2) * H))
        x2 = min(W, int((x + w/2) * W))
        y2 = min(H, int((y + h/2) * H))

        # 确保ROI区域有效
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, W, H

        # 计算全局和 ROI 的 SSIM
        global_ssim = compute_ssim(gt_chunk[i], processed_chunk[i])
        roi_ssim = compute_ssim(
            gt_chunk[i, :, y1:y2, x1:x2],
            processed_chunk[i, :, y1:y2, x1:x2],
        )
        chunk_metrics.append((idx, global_ssim, roi_ssim))
        if idx % 50 == 0:  # 每50帧打印一次，避免输出过多
            print(f"frame{idx}: global={global_ssim:.4f}, roi={roi_ssim:.4f}")
    return chunk_metrics

def calc_ssim(gt: np.ndarray, processed: np.ndarray, roi_xywh: List[Tuple], worker=calc_ssim_worker) -> List[List[Any]]:
    print(f"Calculating SSIM for {gt.shape[0]} frames...")
    print(f"GT shape: {gt.shape}, Processed shape: {processed.shape}")

    # 确保两个数组的形状一致
    min_frames = min(gt.shape[0], processed.shape[0])
    gt = gt[:min_frames]
    processed = processed[:min_frames]

    # 确保ROI数据长度匹配
    roi_xywh = roi_xywh[:min_frames]
    while len(roi_xywh) < min_frames:
        roi_xywh.append((0.5, 0.5, 0.3, 0.3))

    n_frames = gt.shape[0]
    chunk_size = max(1, (n_frames + ssim_calc_thread - 1) // ssim_calc_thread)

    # Prepare bounded-size work chunks for the SSIM workers.
    chunks = []
    for i in range(0, n_frames, chunk_size):
        end_idx = min(i + chunk_size, n_frames)
        chunks.append((
            i,
            gt[i:end_idx],
            processed[i:end_idx],
            roi_xywh[i:end_idx]
        ))

    try:
        # Threads share the already-loaded video arrays.  Passing these chunks to
        # forked workers copies hundreds of megabytes per task and can leave
        # Pool.map waiting forever if a worker is killed under memory pressure.
        with ThreadPoolExecutor(max_workers=min(ssim_calc_thread, len(chunks))) as executor:
            results = list(executor.map(worker, chunks))
    except Exception as e:
        print(f"Error in SSIM worker pool: {e}")
        # Single-thread fallback keeps metric generation available if a custom
        # worker is not thread-safe.
        results = [worker(chunk) for chunk in chunks]

    metric = []
    for chunk_result in results:
        for idx, global_ssim, roi_ssim in chunk_result:
            metric.append([idx, global_ssim, roi_ssim])

    if metric:
        global_avg = sum(m[1] for m in metric) / n_frames
        roi_avg = sum(m[2] for m in metric) / n_frames
        metric.append([-1, global_avg, roi_avg])
    else:
        metric.append([-1, 0.0, 0.0])

    return metric

def read_yolo_roi(folder: str):
    roi_xywh = []
    if not os.path.exists(folder):
        print(f"ROI folder not found: {folder}, using default ROI")
        return []

    try:
        for file in sorted(os.listdir(folder)):
            if not file.endswith('.txt'):
                continue
            with open(os.path.join(folder, file), 'r') as f:
                parts = f.read().strip().split(' ')
            if len(parts) >= 5:
                roi_xywh.append((float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    except Exception as e:
        print(f"Error reading ROI files: {e}")

    if not roi_xywh:
        print("No valid ROI found, using default")
        return []

    return roi_xywh

def _frames_to_nhwc(frames: np.ndarray) -> np.ndarray:
    """Convert cached NCHW RGB frames to the HWC layout expected by encoders."""
    if frames.ndim != 4:
        raise ValueError(f"expected a 4D frame batch, got {frames.shape}")
    if frames.shape[1] == 3:
        return frames.transpose(0, 2, 3, 1)
    if frames.shape[3] == 3:
        return frames
    raise ValueError(f"cannot identify RGB channel axis in {frames.shape}")


def save_video_ffmpeg(frames, save_path):
    """使用FFmpeg保存视频"""
    import subprocess

    frames = _frames_to_nhwc(frames)
    if len(frames) == 0:
        print("No frames to save")
        return

    print(f"Saving {len(frames)} frames to {save_path}")
    height, width = frames.shape[1], frames.shape[2]
    fps = 30.0

    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', f'{width}x{height}',
        '-r', str(fps),
        '-i', '-',
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        save_path
    ]

    process = None
    try:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        for i, frame in enumerate(frames):
            frame_bytes = frame.astype(np.uint8).tobytes()
            process.stdin.write(frame_bytes)
            if i % 100 == 0:
                print(f"Saved {i}/{len(frames)} frames...")

        # communicate() tries to flush stdin, so it cannot be called after
        # stdin.close(). Wait for FFmpeg explicitly instead.
        process.stdin.close()
        process.stdin = None
        returncode = process.wait()
        stderr = process.stderr.read()
        if returncode != 0:
            print(f"✗ FFmpeg error: {stderr.decode(errors='replace')}")
            raise RuntimeError("FFmpeg failed")
        print(f"✓ Video saved to {save_path}")

    except Exception as e:
        print(f"Error saving video: {e}")
        if process is not None and process.poll() is None:
            process.kill()
        if process is not None:
            process.wait()
        if os.path.exists(save_path):
            os.remove(save_path)
        print("Trying OpenCV as fallback...")
        try:
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

            for i, frame in enumerate(frames):
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
                if i % 100 == 0:
                    print(f"OpenCV saved {i}/{len(frames)} frames...")

            out.release()
            print(f"✓ Video saved with OpenCV to {save_path}")
        except Exception as e2:
            print(f"OpenCV also failed: {e2}")

def save_video_with_rebuffer(frames, save_path, rebuffer_times, chunk_duration=3.0):
    """保存包含卡顿效果的视频"""
    frames = _frames_to_nhwc(frames)
    if len(frames) == 0:
        print("No frames to save")
        return

    print(f"Saving {len(frames)} frames with rebuffer effects to {save_path}")

    height, width = frames.shape[1], frames.shape[2]
    fps = 30.0
    frames_per_chunk = int(chunk_duration * fps)

    # 创建包含卡顿的帧序列
    rebuffer_frames = []

    for chunk_idx, rebuffer_time in enumerate(rebuffer_times):
        start_frame = chunk_idx * frames_per_chunk
        end_frame = min((chunk_idx + 1) * frames_per_chunk, len(frames))

        if start_frame >= len(frames):
            break

        # 添加当前chunk的帧
        chunk_frames = frames[start_frame:end_frame]
        rebuffer_frames.extend(chunk_frames)

        # 如果有卡顿，添加静帧
        if rebuffer_time > 0:
            if len(chunk_frames) > 0:
                freeze_frame = chunk_frames[-1]
            else:
                freeze_frame = frames[max(0, start_frame - 1)]

            freeze_frame_count = int(rebuffer_time * fps)
            print(f"Chunk {chunk_idx + 1}: Adding {freeze_frame_count} freeze frames ({rebuffer_time:.3f}s)")

            for _ in range(freeze_frame_count):
                rebuffer_frames.append(freeze_frame)

    # 转换为numpy数组并保存
    final_frames = np.array(rebuffer_frames)
    print(f"Final video: {len(final_frames)} frames (original: {len(frames)}, added: {len(final_frames) - len(frames)})")

    # 使用save_video_ffmpeg保存
    save_video_ffmpeg(final_frames, save_path)

def save_video_with_advanced_rebuffer(frames, save_path, rebuffer_times, chunk_duration=3.0):
    """保存包含高级卡顿效果的视频（带提示）"""
    import cv2

    frames = _frames_to_nhwc(frames)
    if len(frames) == 0:
        print("No frames to save")
        return

    height, width = frames.shape[1], frames.shape[2]
    fps = 30.0
    frames_per_chunk = int(chunk_duration * fps)

    rebuffer_frames = []

    for chunk_idx, rebuffer_time in enumerate(rebuffer_times):
        start_frame = chunk_idx * frames_per_chunk
        end_frame = min((chunk_idx + 1) * frames_per_chunk, len(frames))

        if start_frame >= len(frames):
            break

        # 添加当前chunk的帧
        chunk_frames = frames[start_frame:end_frame]
        rebuffer_frames.extend(chunk_frames)

        # 如果有卡顿，添加带提示的静帧
        if rebuffer_time > 0:
            if len(chunk_frames) > 0:
                freeze_frame = chunk_frames[-1].copy()
            else:
                freeze_frame = frames[max(0, start_frame - 1)].copy()

            # 在卡顿帧上添加"缓冲中..."文字
            freeze_frame_bgr = cv2.cvtColor(freeze_frame, cv2.COLOR_RGB2BGR)

            # 添加半透明背景
            overlay = freeze_frame_bgr.copy()
            cv2.rectangle(overlay, (width//4, height//2 - 30),
                         (3*width//4, height//2 + 30), (0, 0, 0), -1)
            freeze_frame_bgr = cv2.addWeighted(freeze_frame_bgr, 0.7, overlay, 0.3, 0)

            # 添加文字
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"Buffering... {rebuffer_time:.1f}s"
            text_size = cv2.getTextSize(text, font, 1, 2)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height // 2 + text_size[1] // 2
            cv2.putText(freeze_frame_bgr, text, (text_x, text_y),
                       font, 1, (255, 255, 255), 2)

            freeze_frame_with_text = cv2.cvtColor(freeze_frame_bgr, cv2.COLOR_BGR2RGB)

            freeze_frame_count = int(rebuffer_time * fps)
            print(f"Chunk {chunk_idx + 1}: Adding {freeze_frame_count} buffering frames ({rebuffer_time:.3f}s)")

            for _ in range(freeze_frame_count):
                rebuffer_frames.append(freeze_frame_with_text)

    final_frames = np.array(rebuffer_frames)
    save_video_ffmpeg(final_frames, save_path)

def calculate_and_save_averages(dataset: str):
    """
    计算所有客户端的平均值并保存到对应的文件夹中
    Args:
        dataset: 数据集名称
    """
    dataset_dir = os.path.join("client_result", dataset)
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
        return

    # 初始化累加器
    total_performance = {
        'server_delay': [],
        'client_delay': [],
        'trans_delay': [],
        'total_delay': [],
        'rebuffer': [],
        'buffer_level': [],
        'chunk_size': []
    }
    total_quality = {
        'global_ssim': [],
        'roi_ssim': [],
        'QoE': []
    }

    # 遍历所有测试
    for test in datasets[dataset]:
        test_dir = os.path.join(dataset_dir, test)
        performance_file = os.path.join(test_dir, f"{test}_performance.csv")
        quality_file = os.path.join(test_dir, f"{test}_quality.csv")

        # 读取性能指标
        if os.path.exists(performance_file):
            with open(performance_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total_performance['server_delay'].append(float(row['server_delay']))
                    total_performance['client_delay'].append(float(row['client_delay']))
                    total_performance['trans_delay'].append(float(row['trans_delay']))
                    total_performance['total_delay'].append(float(row['total_delay']))
                    total_performance['rebuffer'].append(float(row['rebuffer']))
                    total_performance['buffer_level'].append(float(row['buffer_level']))
                    total_performance['chunk_size'].append(float(row['chunk_size']))

        # 读取质量指标
        if os.path.exists(quality_file):
            with open(quality_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if int(row['idx']) != -1:  # 排除最后的平均行
                        total_quality['global_ssim'].append(float(row['global_ssim']))
                        total_quality['roi_ssim'].append(float(row['roi_ssim']))
                        total_quality['QoE'].append(float(row['QoE']))

    # 计算平均值
    avg_performance = {key: (np.mean(values) if values else 0.0) for key, values in total_performance.items()}
    avg_quality = {key: (np.mean(values) if values else 0.0) for key, values in total_quality.items()}

    # 保存平均值到文件
    avg_performance_file = os.path.join(dataset_dir, "average_performance.csv")
    avg_quality_file = os.path.join(dataset_dir, "average_quality.csv")

    try:
        # 保存性能指标平均值
        with open(avg_performance_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['server_delay', 'client_delay', 'trans_delay', 'total_delay', 'rebuffer', 'buffer_level', 'chunk_size'])
            writer.writerow([avg_performance['server_delay'], avg_performance['client_delay'], avg_performance['trans_delay'],
                             avg_performance['total_delay'], avg_performance['rebuffer'], avg_performance['buffer_level'],
                             avg_performance['chunk_size']])
        print(f"Average performance metrics saved to {avg_performance_file}")

        # 保存质量指标平均值
        with open(avg_quality_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['global_ssim', 'roi_ssim', 'QoE'])
            writer.writerow([avg_quality['global_ssim'], avg_quality['roi_ssim'], avg_quality['QoE']])
        print(f"Average quality metrics saved to {avg_quality_file}")

    except Exception as e:
        print(f"Error saving averages: {e}")

def run_test(dataset: str, name: str) -> Tuple[Dict, List]:
    global net_time
    net_time = 0  # 每个测试重置网络索引

    print(f"Running test: {dataset}/{name}")

    client = client_arch(chunk_len)
    file_info = gen_file_info(dataset, name)

    print(f"File info: {file_info}")

    perform_metric = {
        'arch_delay': [],
        'trans_size': [],
        'trans_delay': [],  # 添加传输时延记录
        'rebuffer': [],
        'buffer': [],  # 新增：记录每个chunk处理后的buffer水平
    }

    try:
        # 初始化
        print("Requesting initialization...")
        init_args = {**client.init_arg(), **client.common_arg(bandwidth=bandwidth)}
        response = requests.post(file_info['server'] + 'init-stream0.m4s', json=init_args)
        response.raise_for_status()
        init_chunk = response.content
        client.receive(init_chunk, is_init=True)
        print("Initialization successful")

        # 处理视频块
        max_chunk = find_max_chunk_number(file_info['local_folder'])
        if max_chunk < 1:
            raise FileNotFoundError(f"no DASH chunks found in {file_info['local_folder']}")
        for chunk_idx in range(1, max_chunk + 1):
            print(f"\nProcessing chunk {chunk_idx}...")
            response = requests.post(
                file_info['server'] + f'chunk-stream0-{chunk_idx:05d}.m4s',
                json=client.common_arg(bandwidth=bandwidth),
            )
            response.raise_for_status()
            received = response.content

            # 解析服务器延迟
            server_delay, = struct.unpack(">f", received[:4])
            timer = time.perf_counter()
            client.receive(received[4:], is_init=False)
            client_delay = time.perf_counter() - timer
            arch_delay = server_delay + client_delay

            # 计算传输相关指标
            trans_size = len(received) - 4
            trans_delay = get_time(trans_size)  # 使用网络trace计算传输时延

            # 计算总延迟
            tot_delay = server_delay + client_delay + trans_delay

            # Update the buffer using the total delay.
            client.buffer -= tot_delay
            if client.buffer < 0:
                rebuffer = -client.buffer
                client.buffer = chunk_len
            else:
                rebuffer = 0
                client.buffer += chunk_len

            # 记录性能指标，包括新增的buffer指标
            perform_metric['arch_delay'].append(arch_delay)
            perform_metric['trans_size'].append(trans_size)
            perform_metric['trans_delay'].append(trans_delay)
            perform_metric['rebuffer'].append(rebuffer)
            perform_metric['buffer'].append(client.buffer)  # 新增：记录当前buffer水平

            # 更新打印信息，包含buffer水平
            print(f"idx={chunk_idx}, server: {server_delay:.3f}s, client: {client_delay:.3f}s, trans: {trans_delay:.3f}s, tot: {tot_delay:.3f}s, rebuffer: {rebuffer:.3f}s, buffer: {client.buffer:.3f}s")

            if client.buffer > buffer_lim:
                time.sleep(client.buffer - buffer_lim)
                client.buffer = buffer_lim

        # 质量评估部分保持不变...
        print("\nCalculating quality metrics...")
        processed_frames = client.cached

        # 转换列表为numpy数组
        # 修复concatenate问题
        if processed_frames and isinstance(processed_frames, list):
            print(f"Converting {len(processed_frames)} processed frames to numpy array...")

            # 检查并统一尺寸
            valid_chunks = [chunk for chunk in processed_frames if len(chunk.shape) >= 3]
            if not valid_chunks:
                return perform_metric, [[-1, 0.0, 0.0, 0.0]]

            # 使用第一个有效chunk的尺寸作为目标
            target_shape = valid_chunks[0].shape[2:4]

            normalized_chunks = []
            for i, chunk in enumerate(processed_frames):
                if len(chunk.shape) >= 3:
                    if chunk.shape[2:4] != target_shape:
                        print(f"Resizing chunk {i} from {chunk.shape[2:4]} to {target_shape}")
                        import cv2
                        resized_chunk = []
                        for frame in chunk:
                            frame_hwc = frame.transpose(1, 2, 0)
                            resized_frame = cv2.resize(frame_hwc, (target_shape[1], target_shape[0]))
                            resized_chunk.append(resized_frame.transpose(2, 0, 1))
                        normalized_chunks.append(np.array(resized_chunk))
                    else:
                        normalized_chunks.append(chunk)

            if normalized_chunks:
                processed = np.concatenate(normalized_chunks, axis=0)
                print(f"Final processed array shape: {processed.shape}")
            else:
                return perform_metric, [[-1, 0.0, 0.0, 0.0]]

        if os.path.exists(file_info['local_gt']):
            print("Loading GT file...")
            gt, fps = decord_video2numpy(file_info['local_gt'])
            roi_xywh = read_yolo_roi(file_info['roi_annotation'])

            print(f"Final GT shape: {gt.shape}")
            print(f"Final processed shape: {processed.shape}")

            # 确保两个数组的格式匹配
            if len(gt.shape) >= 3 and len(processed.shape) >= 3:
                if gt.shape[2:4] != processed.shape[2:4]:
                    print(f"Shape mismatch: GT{gt.shape[2:4]} vs Processed{processed.shape[2:4]}")
                    print("Resizing processed frames to match GT...")
                    import cv2
                    resized_processed = []
                    target_h, target_w = gt.shape[2], gt.shape[3]
                    for frame in processed:
                        if frame.shape[1:3] != (target_h, target_w):
                            frame_hwc = frame.transpose(1, 2, 0)
                            resized_frame = cv2.resize(frame_hwc, (target_w, target_h))
                            resized_processed.append(resized_frame.transpose(2, 0, 1))
                        else:
                            resized_processed.append(frame)
                    processed = np.array(resized_processed)
                    print(f"Resized processed shape: {processed.shape}")

            quality_metric = calc_ssim(gt, processed, roi_xywh)
            print(f"\nOverall quality: global={quality_metric[-1][1]:.4f}, roi={quality_metric[-1][2]:.4f}")
        else:
            print(f"GT file not found: {file_info['local_gt']}")
            quality_metric = [[-1, 0.0, 0.0]]

        # 计算 QoE
        print("Calculating QoE...")
        if len(quality_metric) > 1:
            video_duration = test_len  # 使用测试视频的总时长（24秒）
            qoe_value = calc_qoe(quality_metric[:-1], perform_metric['rebuffer'], video_duration)
            print(f"QoE: {qoe_value:.4f}")

            # 为每个帧数据添加QoE值
            for i, metric_row in enumerate(quality_metric):
                if i < len(quality_metric) - 1:
                    metric_row.append(qoe_value)
                else:
                    metric_row.append(qoe_value)
        else:
            print("Insufficient data for QoE calculation")
            for metric_row in quality_metric:
                metric_row.append(0.0)

        # 在run_test函数的视频保存部分，将变量定义移到try块外面
        print("\nSaving processed videos...")
        os.makedirs(os.path.dirname(file_info['video_save']), exist_ok=True)

        if processed_frames and isinstance(processed_frames, list):
            base_path = file_info['video_save'].replace('.mp4', '')

            # 将变量定义移到这里，在try块外面
            original_path = base_path + '_original.mp4'
            simple_rebuffer_path = base_path + '_with_rebuffer.mp4'
            advanced_rebuffer_path = base_path + '_advanced_rebuffer.mp4'

            print(f"✓ Videos saved to {os.path.dirname(file_info['video_save'])}:")

            try:
                # 1. 保存原始视频（无卡顿）
                save_video_ffmpeg(processed, original_path)
                print(f"  - Original (no rebuffer): {original_path}")

                # # 2. 保存简单卡顿版本
                # save_video_with_rebuffer(processed, simple_rebuffer_path,
                #                        perform_metric['rebuffer'], chunk_len)
                # print(f"  - Simple rebuffer: {simple_rebuffer_path}")

                # # 3. 保存高级卡顿版本（带提示）
                # save_video_with_advanced_rebuffer(processed, advanced_rebuffer_path,
                #                                 perform_metric['rebuffer'], chunk_len)
                # print(f"  - Advanced rebuffer: {advanced_rebuffer_path}")

                # # 4. 将高级版本复制为主文件
                # import shutil
                # shutil.copy2(advanced_rebuffer_path, file_info['video_save'])
                # print(f"  - Main file: {file_info['video_save']}")

            except Exception as video_error:
                print(f"Error saving additional videos: {video_error}")
                # 至少保存基本版本
                save_video_ffmpeg(processed, file_info['video_save'])
                print(f"  - Basic video: {file_info['video_save']}")
        return perform_metric, quality_metric

    except Exception as e:
        print(f"Error in run_test: {e}")
        import traceback
        traceback.print_exc()
        raise

def run_dataset(dataset: str):
    print(f"Running dataset {dataset}...")
    dataset_dir = os.path.join("client_result", dataset)

    for test in datasets[dataset]:
        perform_metric, quality_metric = run_test(dataset, test)

        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        # 创建测试专用目录
        test_dir = os.path.join(dataset_dir, test)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        # 保存质量结果 - 确保有有效数据
        quality_file = os.path.join(test_dir, f"{test}_quality.csv")
        try:
            with open(quality_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['idx', 'global_ssim', 'roi_ssim', 'QoE'])

                # 检查是否有有效的质量数据
                if len(quality_metric) > 1 and quality_metric[0][0] != -1:
                    # 有效数据，保存所有行
                    for data in quality_metric:
                        writer.writerow(data)
                    print(f"Quality results saved to {quality_file} ({len(quality_metric)} rows)")
                else:
                    # 没有有效数据，只保存默认行
                    writer.writerow([-1, 0.0, 0.0, 0.0])
                    print(f"Default quality results saved to {quality_file} (no valid data)")

        except Exception as e:
            print(f"Error saving quality results: {e}")

        # 保存性能结果
        performance_file = os.path.join(test_dir, f"{test}_performance.csv")
        try:
            with open(performance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['chunk_idx', 'server_delay', 'client_delay', 'trans_delay', 'total_delay', 'rebuffer', 'buffer_level', 'chunk_size'])

                for i in range(len(perform_metric['arch_delay'])):
                    # 计算客户端延迟 (arch_delay包含server+client)
                    server_delay = perform_metric['arch_delay'][i] - (perform_metric.get('client_delay', [0] * len(perform_metric['arch_delay']))[i] if 'client_delay' in perform_metric else 0)

                    writer.writerow([
                        i+1,
                        server_delay,
                        perform_metric.get('client_delay', [0] * len(perform_metric['arch_delay']))[i] if 'client_delay' in perform_metric else 0,
                        perform_metric['trans_delay'][i],
                        perform_metric['arch_delay'][i] + perform_metric['trans_delay'][i],
                        perform_metric['rebuffer'][i],
                        perform_metric['buffer'][i] if 'buffer' in perform_metric else 0,
                        perform_metric['trans_size'][i],  # 添加chunk_size（传输块大小）
                    ])
            print(f"Performance metrics saved to {performance_file}")

        except Exception as e:
            print(f"Error saving performance metrics: {e}")

        # 打印性能统计
        print(f"\nPerformance Summary for {test}:")
        print(f"  Average server delay: {np.mean(perform_metric['arch_delay']):.3f}s")
        print(f"  Average transmission size: {np.mean(perform_metric['trans_size']):.0f} bytes")
        print(f"  Average transmission delay: {np.mean(perform_metric['trans_delay']):.3f}s")
        print(f"  Total rebuffer time: {sum(perform_metric['rebuffer']):.3f}s")
        if 'buffer' in perform_metric:
            print(f"  Average buffer level: {np.mean(perform_metric['buffer']):.3f}s")
            print(f"  Final buffer level: {perform_metric['buffer'][-1]:.3f}s")

        # 打印质量统计
        if len(quality_metric) > 1 and quality_metric[-1][0] == -1:
            final_qoe = quality_metric[-1][3] if len(quality_metric[-1]) > 3 else 0.0
            print(f"  Final QoE: {final_qoe:.4f}")
        else:
            print(f"  Final QoE: 0.0000")

def run_client(dataset: str):
    """运行单个客户端"""
    print(f"Starting client for dataset: {dataset}")
    run_dataset(dataset)
    print(f"Client for dataset {dataset} completed.")

def main():
    print("Starting basic evaluation...")

    client_datasets = list(datasets)
    for dataset in client_datasets:
        run_client(dataset)

    print("All clients completed!")

    # 计算并保存平均值
    for dataset in set(client_datasets):  # 去重，避免重复计算
        calculate_and_save_averages(dataset)

if __name__ == '__main__':
    main()
