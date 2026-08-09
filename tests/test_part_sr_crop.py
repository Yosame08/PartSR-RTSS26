import unittest

import torch

from edge_arch.edge_PartSR import crop_rois


class PartSRCropTest(unittest.TestCase):
    def test_vectorized_crop_matches_per_frame_slices_and_preserves_channels(self):
        frames = torch.arange(4 * 3 * 8 * 9).reshape(4, 3, 8, 9)
        boxes = torch.tensor([
            [0, 0, 4, 3],
            [2, 1, 6, 4],
            [5, 4, 9, 7],
            [1, 5, 5, 8],
        ])

        expected = torch.stack([
            frames[index, :, y1:y2, x1:x2]
            for index, (x1, y1, x2, y2) in enumerate(boxes.tolist())
        ])

        self.assertTrue(torch.equal(crop_rois(frames, boxes), expected))


if __name__ == "__main__":
    unittest.main()
