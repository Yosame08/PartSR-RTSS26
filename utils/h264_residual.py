import struct
import zlib

import numpy as np


H264_RESIDUAL_MAGIC = b"H2RS"
H264_RESIDUAL_VERSION = 1
H264_RESIDUAL_HEADER_SIZE = 80
H264_RESIDUAL_METRIC_NONZERO_LUMA = 1
H264_RESIDUAL_NOPTS = -(1 << 63)

_H264_RESIDUAL_HEADER = struct.Struct("<4sHHIIIIIIIIIiIQqqI")
_H264_RESIDUAL_KNOWN_FLAGS = 0x1F


def parse_h264_residual(data: bytes, expected_frames: int) -> np.ndarray:
    """Parse H2RS v1 records into a flat (frame, macroblock) uint32 array."""
    if not isinstance(data, bytes):
        raise TypeError(f"data must be bytes, got {type(data).__name__}")
    if expected_frames <= 0:
        raise ValueError(f"expected_frames must be positive, got {expected_frames}")

    residuals = []
    expected_shape = None
    previous_timestamp = None
    offset = 0

    while offset < len(data):
        if len(data) - offset < H264_RESIDUAL_HEADER_SIZE:
            raise ValueError(f"partial H2RS header at byte {offset}")

        header = data[offset:offset + H264_RESIDUAL_HEADER_SIZE]
        (
            magic,
            version,
            header_size,
            record_size,
            payload_size,
            payload_crc,
            metric,
            flags,
            mb_width,
            mb_height,
            frame_width,
            frame_height,
            _poc,
            _frame_num,
            sequence,
            pts,
            best_effort_timestamp,
            header_crc,
        ) = _H264_RESIDUAL_HEADER.unpack(header)

        if magic != H264_RESIDUAL_MAGIC:
            raise ValueError(f"invalid H2RS magic at byte {offset}: {magic!r}")
        if version != H264_RESIDUAL_VERSION:
            raise ValueError(f"unsupported H2RS version {version} at record {sequence}")
        if header_size != H264_RESIDUAL_HEADER_SIZE:
            raise ValueError(f"invalid H2RS header size {header_size} at record {sequence}")
        if zlib.crc32(header[:76]) & 0xFFFFFFFF != header_crc:
            raise ValueError(f"H2RS header CRC mismatch at record {sequence}")
        if metric != H264_RESIDUAL_METRIC_NONZERO_LUMA:
            raise ValueError(f"unsupported H2RS metric {metric} at record {sequence}")
        if flags & ~_H264_RESIDUAL_KNOWN_FLAGS:
            raise ValueError(f"unknown H2RS flags 0x{flags:x} at record {sequence}")
        if sequence != len(residuals):
            raise ValueError(
                f"non-contiguous H2RS sequence {sequence}; expected {len(residuals)}"
            )
        if mb_width <= 0 or mb_height <= 0 or frame_width <= 0 or frame_height <= 0:
            raise ValueError(f"invalid H2RS dimensions at record {sequence}")
        if mb_width * 16 < frame_width or mb_height * 16 < frame_height:
            raise ValueError(f"H2RS frame exceeds its macroblock grid at record {sequence}")

        expected_payload_size = mb_width * mb_height * np.dtype("<u4").itemsize
        if payload_size != expected_payload_size:
            raise ValueError(
                f"invalid H2RS payload size {payload_size} at record {sequence}; "
                f"expected {expected_payload_size}"
            )
        if record_size != H264_RESIDUAL_HEADER_SIZE + payload_size:
            raise ValueError(f"invalid H2RS record size {record_size} at record {sequence}")

        end = offset + record_size
        if end > len(data):
            raise ValueError(f"partial H2RS payload at record {sequence}")
        payload = data[offset + H264_RESIDUAL_HEADER_SIZE:end]
        if zlib.crc32(payload) & 0xFFFFFFFF != payload_crc:
            raise ValueError(f"H2RS payload CRC mismatch at record {sequence}")

        shape = (mb_height, mb_width)
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(
                f"H2RS macroblock grid changed from {expected_shape} to {shape} "
                f"at record {sequence}"
            )

        values = np.frombuffer(payload, dtype="<u4")
        if values.size and int(values.max()) > 256:
            raise ValueError(f"H2RS luma count exceeds 256 at record {sequence}")
        residuals.append(values)

        has_pts = bool(flags & 1)
        has_best_effort_timestamp = bool(flags & 2)
        if has_pts != (pts != H264_RESIDUAL_NOPTS):
            raise ValueError(f"inconsistent H2RS pts flag at record {sequence}")
        if has_best_effort_timestamp != (
            best_effort_timestamp != H264_RESIDUAL_NOPTS
        ):
            raise ValueError(
                f"inconsistent H2RS best-effort timestamp flag at record {sequence}"
            )
        timestamp = (
            best_effort_timestamp
            if has_best_effort_timestamp
            else pts if has_pts else None
        )
        if (
            timestamp is not None
            and previous_timestamp is not None
            and timestamp < previous_timestamp
        ):
            raise ValueError(f"H2RS timestamps are not ordered at record {sequence}")
        if timestamp is not None:
            previous_timestamp = timestamp

        offset = end

    if len(residuals) != expected_frames:
        raise ValueError(
            f"H2RS frame count mismatch: decoded {len(residuals)}, "
            f"expected {expected_frames}"
        )

    return np.stack(residuals)
