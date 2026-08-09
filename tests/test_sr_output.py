import unittest
from unittest.mock import mock_open, patch

import numpy as np
import torch

from utils.sr_output import sr_tensor_to_uint8_rgb


class SROutputTest(unittest.TestCase):
    def test_clamps_and_preserves_existing_uint8_truncation(self):
        frames = torch.tensor(
            [[[[-0.25, 0.0, 0.5, 1.0, 1.25]]]], dtype=torch.float32
        ).repeat(1, 3, 1, 1)

        result = sr_tensor_to_uint8_rgb(frames)

        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.array_equal(result[0, 0, 0], [0, 0, 127, 255, 255]))

    def test_is_byte_exact_with_legacy_clamp_mul_byte_conversion(self):
        frames = torch.linspace(-1.5, 2.0, steps=120, dtype=torch.float32)
        frames = frames.reshape(2, 3, 4, 5).transpose(2, 3)
        expected = (
            frames.detach()
            .clamp(0, 1)
            .mul(255)
            .byte()
            .cpu()
            .contiguous()
            .numpy()
        )

        result = sr_tensor_to_uint8_rgb(frames)

        self.assertEqual(result.tobytes(), expected.tobytes())

    def test_preserves_nchw_rgb_layout(self):
        frames = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
        frames[0, 0, 0, 1] = 1.0
        frames[0, 1, 1, 0] = 0.5
        frames[0, 2, 1, 1] = 0.25

        result = sr_tensor_to_uint8_rgb(frames)

        self.assertEqual(result.shape, (1, 3, 2, 2))
        self.assertTrue(result.flags.c_contiguous)
        self.assertEqual(result[0, 0, 0, 1], 255)
        self.assertEqual(result[0, 1, 1, 0], 127)
        self.assertEqual(result[0, 2, 1, 1], 63)
        self.assertEqual(result[0, 2, 0, 1], 0)

    def test_ffmpeg_encoder_receives_nhwc_rgb24_bytes(self):
        from utils.utils import ffmpeg_tensor_to_bytes

        frames = torch.tensor(
            [[
                [[0.0, 1.0]],
                [[0.5, 0.25]],
                [[1.0, 0.0]],
            ]],
            dtype=torch.float32,
        )

        with patch("utils.utils.subprocess.Popen") as popen, patch(
            "builtins.open", mock_open(read_data=b"encoded")
        ):
            process = popen.return_value
            process.returncode = 0
            process.communicate.return_value = (b"", b"")

            encoded = ffmpeg_tensor_to_bytes(frames, 30.0, "sr-output-layout-test")

        self.assertEqual(encoded, b"encoded")
        self.assertEqual(
            process.communicate.call_args.kwargs["input"],
            bytes([0, 127, 255, 255, 63, 0]),
        )

    def test_does_not_modify_input_or_attach_numpy_to_autograd(self):
        frames = torch.tensor(
            [[[[1.2]], [[0.5]], [[-0.2]]]], dtype=torch.float32, requires_grad=True
        )
        original = frames.detach().clone()

        result = sr_tensor_to_uint8_rgb(frames)

        self.assertTrue(torch.equal(frames.detach(), original))
        self.assertTrue(np.array_equal(result.reshape(-1), [255, 127, 0]))

    def test_rejects_nan_and_infinity(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                frames = torch.zeros((1, 3, 1, 1), dtype=torch.float32)
                frames[0, 0, 0, 0] = value
                with self.assertRaisesRegex(ValueError, "NaN or infinite"):
                    sr_tensor_to_uint8_rgb(frames)

    def test_rejects_invalid_type_shape_channels_and_dtype(self):
        invalid_cases = [
            (np.zeros((1, 3, 1, 1), dtype=np.float32), TypeError),
            (torch.zeros((3, 1, 1), dtype=torch.float32), ValueError),
            (torch.zeros((1, 1, 1, 1), dtype=torch.float32), ValueError),
            (torch.zeros((0, 3, 1, 1), dtype=torch.float32), ValueError),
            (torch.zeros((1, 3, 1, 1), dtype=torch.uint8), TypeError),
        ]

        for frames, expected_error in invalid_cases:
            with self.subTest(frames=type(frames).__name__, shape=frames.shape):
                with self.assertRaises(expected_error):
                    sr_tensor_to_uint8_rgb(frames)


if __name__ == "__main__":
    unittest.main()
