import unittest

import numpy as np
import torch

from extractor.extractor import RoIExtractor


class ExtractorDynamicityTest(unittest.TestCase):
    def test_full_frame_roi_has_no_outside_dynamicity(self):
        extractor = RoIExtractor(
            tensors=torch.zeros(1, 3, 320, 180),
            ndarray=np.zeros((1, 320, 180, 3), dtype=np.uint8),
            types=["I"],
            mvs=[[]],
            framerate=30.0,
            residual_arr=np.zeros((1, 1), dtype=np.float32),
            detector=None,
        )

        self.assertEqual(extractor.calc_dyn_exclusive(0, (0, 0, 180, 320)), 0.0)
