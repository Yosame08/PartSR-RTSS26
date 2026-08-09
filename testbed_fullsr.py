import argparse
import csv
import json
import struct
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import requests
import torch  # Load Torch first to avoid the Decord native runtime's libstdc++ random_device conflict.
import decord
from skimage.metrics import structural_similarity

from client_arch.client import Client
from config import config


NETWORK_TRACE_KBPS = [
    11256, 13946, 12408, 13777, 11297, 7744, 13324, 4796, 7356,
    11768, 13922, 12629, 11842, 16473, 14448, 18178, 16397, 15147,
    11007, 9258, 9854, 6110, 11633, 12596, 6258,
]


def simulated_transfer_seconds(size_bytes: int, trace_offset: int) -> Tuple[float, int]:
    remaining = size_bytes
    elapsed = 0.0
    offset = trace_offset
    while remaining > 0:
        bytes_per_second = NETWORK_TRACE_KBPS[offset % len(NETWORK_TRACE_KBPS)] * 128
        offset += 1
        if remaining <= bytes_per_second:
            elapsed += remaining / bytes_per_second
            break
        remaining -= bytes_per_second
        elapsed += 1.0
    return elapsed, offset


def read_yolo_rois(folder: Path, expected_frames: int) -> List[Tuple[float, float, float, float]]:
    files = sorted(folder.glob('*.txt'))
    if len(files) != expected_frames:
        raise ValueError(f"expected {expected_frames} ROI files, found {len(files)} in {folder}")

    rois = []
    for path in files:
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"empty ROI annotation: {path}")
        fields = lines[0].split()
        if len(fields) < 5:
            raise ValueError(f"invalid YOLO annotation: {path}")
        x, y, width, height = map(float, fields[1:5])
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"out-of-range YOLO annotation: {path}")
        rois.append((x, y, width, height))
    return rois


def roi_bounds(roi: Sequence[float], height: int, width: int) -> Tuple[int, int, int, int]:
    x, y, roi_width, roi_height = roi
    x1 = max(0, int((x - roi_width / 2) * width))
    y1 = max(0, int((y - roi_height / 2) * height))
    x2 = min(width, int((x + roi_width / 2) * width))
    y2 = min(height, int((y + roi_height / 2) * height))
    if x2 - x1 < 7 or y2 - y1 < 7:
        raise ValueError(f"ROI is too small for SSIM: {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


def frame_metrics(
    gt: np.ndarray,
    processed: np.ndarray,
    roi: Sequence[float],
) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    if gt.shape != processed.shape or gt.ndim != 3 or gt.shape[0] != 3:
        raise ValueError(f"expected equal CHW RGB frames, got gt={gt.shape}, processed={processed.shape}")
    if gt.dtype != np.uint8 or processed.dtype != np.uint8:
        raise ValueError(f"expected uint8 frames, got gt={gt.dtype}, processed={processed.dtype}")

    _, height, width = gt.shape
    x1, y1, x2, y2 = roi_bounds(roi, height, width)
    global_ssim = structural_similarity(gt, processed, channel_axis=0, data_range=255)
    roi_ssim = structural_similarity(
        gt[:, y1:y2, x1:x2],
        processed[:, y1:y2, x1:x2],
        channel_axis=0,
        data_range=255,
    )
    swapped_ssim = structural_similarity(gt, processed[[2, 1, 0]], channel_axis=0, data_range=255)

    difference = np.abs(processed.astype(np.int16) - gt.astype(np.int16))
    channel_mae_sum = difference.sum(axis=(1, 2), dtype=np.float64)
    gt_channel_sum = gt.sum(axis=(1, 2), dtype=np.float64)
    processed_channel_sum = processed.sum(axis=(1, 2), dtype=np.float64)
    return global_ssim, roi_ssim, swapped_ssim, channel_mae_sum, gt_channel_sum, processed_channel_sum


def resolve_data_paths(data_root: Path, dataset: str, clip: str) -> Tuple[Path, Path]:
    clip_root = data_root / dataset / clip
    dataset_name = dataset.replace('_3s_720p', '')
    return (
        clip_root / f"720p-{dataset_name}-{clip}.mp4",
        clip_root / f"{clip}_annotate" / "labels",
    )


def git_commit() -> str:
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def run(args: argparse.Namespace) -> Dict:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_path, roi_path = resolve_data_paths(args.data_root.resolve(), args.dataset, args.clip)
    if not gt_path.is_file():
        raise FileNotFoundError(gt_path)
    if not roi_path.is_dir():
        raise FileNotFoundError(roi_path)

    frames_per_chunk = config['chunk_len'] * 30
    expected_frames = args.chunks * frames_per_chunk
    rois = read_yolo_rois(roi_path, expected_frames)
    gt_reader = decord.VideoReader(str(gt_path), ctx=decord.cpu(0), num_threads=1)
    if len(gt_reader) != expected_frames:
        raise ValueError(f"expected {expected_frames} GT frames, found {len(gt_reader)} in {gt_path}")

    client = Client(config['chunk_len'])
    base_url = f"{args.edge_server.rstrip('/')}/{args.source_dataset}/{args.clip}"
    session = requests.Session()
    init_response = session.post(
        f"{base_url}/init-stream0.m4s",
        json={**client.init_arg(), **client.common_arg()},
        timeout=args.request_timeout,
    )
    init_response.raise_for_status()
    client.receive(init_response.content, is_init=True)

    quality_rows = []
    performance_rows = []
    trace_offset = 0
    channel_mae_sum = np.zeros(3, dtype=np.float64)
    gt_channel_sum = np.zeros(3, dtype=np.float64)
    processed_channel_sum = np.zeros(3, dtype=np.float64)
    pixels_per_channel = 0

    for chunk_idx in range(1, args.chunks + 1):
        request_start = time.perf_counter()
        response = session.post(
            f"{base_url}/chunk-stream0-{chunk_idx:05d}.m4s",
            json=client.common_arg(),
            timeout=args.request_timeout,
        )
        round_trip = time.perf_counter() - request_start
        response.raise_for_status()
        if len(response.content) <= 4:
            raise ValueError(f"chunk {chunk_idx} response is too short: {len(response.content)} bytes")
        server_delay, = struct.unpack('>f', response.content[:4])

        client_start = time.perf_counter()
        client.receive(response.content[4:], is_init=False)
        client_delay = time.perf_counter() - client_start
        if len(client.cached) != 1:
            raise RuntimeError(f"expected one cached chunk, found {len(client.cached)}")
        processed = client.cached.pop()
        if processed.shape[0] != frames_per_chunk:
            raise ValueError(
                f"chunk {chunk_idx} has {processed.shape[0]} frames, expected {frames_per_chunk}"
            )

        frame_start = (chunk_idx - 1) * frames_per_chunk
        frame_stop = frame_start + frames_per_chunk
        gt = gt_reader.get_batch(range(frame_start, frame_stop)).asnumpy().transpose(0, 3, 1, 2)
        if gt.shape != processed.shape:
            raise ValueError(f"chunk {chunk_idx} shape mismatch: gt={gt.shape}, processed={processed.shape}")

        for local_idx in range(frames_per_chunk):
            frame_idx = frame_start + local_idx
            metrics = frame_metrics(gt[local_idx], processed[local_idx], rois[frame_idx])
            global_ssim, roi_ssim, swapped_ssim, mae_sum, gt_sum, processed_sum = metrics
            quality_rows.append((frame_idx, global_ssim, roi_ssim, swapped_ssim))
            channel_mae_sum += mae_sum
            gt_channel_sum += gt_sum
            processed_channel_sum += processed_sum
            pixels_per_channel += gt.shape[2] * gt.shape[3]

        payload_bytes = len(response.content) - 4
        transfer_delay, trace_offset = simulated_transfer_seconds(payload_bytes, trace_offset)
        performance_rows.append({
            'chunk_idx': chunk_idx,
            'server_delay_s': server_delay,
            'client_delay_s': client_delay,
            'http_round_trip_s': round_trip,
            'payload_bytes': payload_bytes,
            'simulated_transfer_delay_s': transfer_delay,
            'total_simulated_delay_s': server_delay + client_delay + transfer_delay,
            'frames': processed.shape[0],
        })
        print(
            f"chunk={chunk_idx}/{args.chunks} frames={processed.shape[0]} "
            f"server={server_delay:.3f}s client={client_delay:.3f}s "
            f"http={round_trip:.3f}s payload={payload_bytes}B"
        )

    quality_path = output_dir / 'quality.csv'
    with quality_path.open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['frame_idx', 'global_ssim', 'roi_ssim', 'rb_swapped_global_ssim'])
        writer.writerows(quality_rows)

    performance_path = output_dir / 'performance.csv'
    with performance_path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(performance_rows[0]))
        writer.writeheader()
        writer.writerows(performance_rows)

    global_values = np.array([row[1] for row in quality_rows])
    roi_values = np.array([row[2] for row in quality_rows])
    swapped_values = np.array([row[3] for row in quality_rows])
    summary = {
        'git_commit': git_commit(),
        'dataset': args.dataset,
        'source_dataset': args.source_dataset,
        'clip': args.clip,
        'chunks': args.chunks,
        'frames': len(quality_rows),
        'edge_server': args.edge_server,
        'gt_video': str(gt_path),
        'roi_dir': str(roi_path),
        'global_ssim': {
            'mean': float(global_values.mean()),
            'min': float(global_values.min()),
            'max': float(global_values.max()),
        },
        'roi_ssim': {
            'mean': float(roi_values.mean()),
            'min': float(roi_values.min()),
            'max': float(roi_values.max()),
        },
        'rb_swapped_global_ssim': {
            'mean': float(swapped_values.mean()),
            'min': float(swapped_values.min()),
            'max': float(swapped_values.max()),
        },
        'color_order_margin': float(global_values.mean() - swapped_values.mean()),
        'channel_order': 'RGB',
        'frame_layout': 'NCHW',
        'dtype': 'uint8',
        'gt_channel_mean_rgb': (gt_channel_sum / pixels_per_channel).tolist(),
        'processed_channel_mean_rgb': (processed_channel_sum / pixels_per_channel).tolist(),
        'channel_mae_rgb': (channel_mae_sum / pixels_per_channel).tolist(),
        'minimum_required_global_ssim': args.min_ssim,
        'network_trace_kbps': NETWORK_TRACE_KBPS,
        'config': {
            'chunk_len': config['chunk_len'],
            'video_width': config['video_width'],
            'video_height': config['video_height'],
        },
    }
    summary['ssim_valid'] = summary['global_ssim']['mean'] >= args.min_ssim
    summary['color_order_valid'] = (
        summary['rb_swapped_global_ssim']['mean'] <= summary['global_ssim']['mean'] + 0.05
    )
    summary['validation_passed'] = summary['ssim_valid'] and summary['color_order_valid']
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    if not summary['color_order_valid']:
        raise RuntimeError(
            "R/B-swapped output matches GT materially better; likely RGB/BGR conversion bug"
        )
    if not summary['ssim_valid']:
        raise RuntimeError(
            f"mean global SSIM {summary['global_ssim']['mean']:.4f} is below {args.min_ssim:.4f}"
        )
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Strict EdgeFullSR end-to-end validation')
    parser.add_argument('--dataset', default='sell_3s_720p')
    parser.add_argument('--source-dataset')
    parser.add_argument('--clip', default='plant')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--edge-server', default=config['edge_server'])
    parser.add_argument('--chunks', type=int, default=8)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--min-ssim', type=float, default=0.5)
    parser.add_argument('--request-timeout', type=float, default=600)
    args = parser.parse_args()
    if args.source_dataset is None:
        args.source_dataset = args.dataset
    if args.chunks <= 0:
        parser.error('--chunks must be positive')
    if not 0 <= args.min_ssim <= 1:
        parser.error('--min-ssim must be between 0 and 1')
    return args


if __name__ == '__main__':
    run(parse_args())
