import os
import struct
import subprocess
import unittest
import zlib
from unittest.mock import patch

import numpy as np

from utils.h264_residual import parse_h264_residual


HEADER = struct.Struct("<4sHHIIIIIIIIIiIQqqI")
NOPTS = -(1 << 63)


def make_record(sequence, values, pts=None, best_effort_timestamp=None):
    values = np.asarray(values, dtype="<u4")
    mb_height, mb_width = values.shape
    payload = values.tobytes()
    flags = (1 if pts is not None else 0) | (
        2 if best_effort_timestamp is not None else 0
    )
    fields = (
        b"H2RS",
        1,
        HEADER.size,
        HEADER.size + len(payload),
        len(payload),
        zlib.crc32(payload) & 0xFFFFFFFF,
        1,
        flags,
        mb_width,
        mb_height,
        mb_width * 16,
        mb_height * 16,
        sequence * 2,
        sequence,
        sequence,
        pts if pts is not None else NOPTS,
        best_effort_timestamp if best_effort_timestamp is not None else NOPTS,
        0,
    )
    header = HEADER.pack(*fields)
    fields = (*fields[:-1], zlib.crc32(header[:76]) & 0xFFFFFFFF)
    return HEADER.pack(*fields) + payload


class H264ResidualProtocolTest(unittest.TestCase):
    def test_parses_ordered_uint32_records(self):
        records = np.array(
            [
                [[0, 1], [255, 256]],
                [[4, 3], [2, 1]],
            ],
            dtype=np.uint32,
        )
        data = b"".join(
            make_record(index, values, pts=index, best_effort_timestamp=index)
            for index, values in enumerate(records)
        )
        expected = records.reshape(2, 4)

        result = parse_h264_residual(data, expected_frames=2)

        self.assertEqual(result.dtype, np.dtype("<u4"))
        self.assertTrue(np.array_equal(result, expected))

    def test_rejects_corrupt_or_incompatible_records(self):
        valid = make_record(0, [[1, 2], [3, 4]], pts=0)
        cases = {
            "partial header": valid[:20],
            "header CRC": valid[:10] + bytes([valid[10] ^ 1]) + valid[11:],
            "payload CRC": valid[:-1] + bytes([valid[-1] ^ 1]),
            "sequence": make_record(1, [[1]], pts=0),
            "metric": bytearray(valid),
            "count": make_record(0, [[257]], pts=0),
        }
        struct.pack_into("<I", cases["metric"], 20, 2)
        struct.pack_into(
            "<I",
            cases["metric"],
            76,
            zlib.crc32(cases["metric"][:76]) & 0xFFFFFFFF,
        )

        for name, data in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    parse_h264_residual(bytes(data), expected_frames=1)

    def test_rejects_wrong_frame_count_and_timestamp_order(self):
        record = make_record(0, [[1]], pts=1)
        with self.assertRaisesRegex(ValueError, "frame count mismatch"):
            parse_h264_residual(record, expected_frames=2)

        data = record + make_record(1, [[2]], pts=0)
        with self.assertRaisesRegex(ValueError, "timestamps are not ordered"):
            parse_h264_residual(data, expected_frames=2)

    def test_ffmpeg_uses_dedicated_fd_and_strict_frame_limit(self):
        from utils import utils

        data = make_record(0, [[1, 2], [3, 4]], pts=0)
        captured_command = None

        def run(command, **kwargs):
            nonlocal captured_command
            captured_command = command
            residual_fd, = kwargs["pass_fds"]
            os.write(residual_fd, data)
            return subprocess.CompletedProcess(command, 0, stderr=b"")

        with patch("utils.utils.subprocess.run", side_effect=run):
            result = utils.ffmpeg_decode_residual("input.mp4", frame_num=1)

        self.assertTrue(np.array_equal(result, [[1, 2, 3, 4]]))
        self.assertLess(captured_command.index("-c:v"), captured_command.index("-residual_fd"))
        self.assertLess(captured_command.index("-residual_fd"), captured_command.index("-i"))
        self.assertEqual(
            captured_command[captured_command.index("-frames:v") + 1],
            "1",
        )
        self.assertNotIn("-threads", captured_command)


if __name__ == "__main__":
    unittest.main()
