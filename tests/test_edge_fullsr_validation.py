import unittest

import numpy as np

from edge_arch.edge_FullSR import validate_lr_frames
from testbed_fullsr import frame_metrics, roi_bounds, simulated_transfer_seconds


class EdgeFullSRValidationTest(unittest.TestCase):
    def test_identical_rgb_frame_scores_one(self):
        frame = np.zeros((3, 16, 16), dtype=np.uint8)
        frame[0, :, :8] = 255
        frame[2, :, 8:] = 64
        global_ssim, roi_ssim, swapped_ssim, *_ = frame_metrics(
            frame,
            frame.copy(),
            (0.5, 0.5, 1.0, 1.0),
        )
        self.assertAlmostEqual(global_ssim, 1.0)
        self.assertAlmostEqual(roi_ssim, 1.0)
        self.assertLess(swapped_ssim, global_ssim)

    def test_shape_mismatch_fails(self):
        with self.assertRaises(ValueError):
            frame_metrics(
                np.zeros((3, 16, 16), dtype=np.uint8),
                np.zeros((16, 16, 3), dtype=np.uint8),
                (0.5, 0.5, 1.0, 1.0),
            )

    def test_roi_too_small_fails(self):
        with self.assertRaises(ValueError):
            roi_bounds((0.5, 0.5, 0.001, 0.001), 720, 1280)

    def test_network_trace_consumes_bytes(self):
        elapsed, offset = simulated_transfer_seconds(100, 0)
        self.assertGreater(elapsed, 0)
        self.assertEqual(offset, 1)

    def test_edge_full_rejects_high_resolution_source(self):
        with self.assertRaisesRegex(ValueError, 'expected NCHW low-resolution frames'):
            validate_lr_frames(np.zeros((90, 3, 1280, 720), dtype=np.uint8))

    def test_edge_full_accepts_configured_low_resolution_source(self):
        validate_lr_frames(np.zeros((90, 3, 320, 180), dtype=np.uint8))


if __name__ == '__main__':
    unittest.main()
