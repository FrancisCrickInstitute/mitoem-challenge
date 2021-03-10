import unittest

import torch

from src.ml.loss import iou, dice, dice_loss


class TestModelA(unittest.TestCase):

    def setUp(self):
        self.a = torch.tensor([
            [1., 0., 1.],
            [0., 1., 0.],
            [0., 0., 1.],
        ])
        self.b = torch.tensor([
            [0., 1., 1.],
            [0., 1., 0.],
            [1., 0., 0.],
        ]).T

    def test_iou(self):
        # intersection = 2
        # union = 6
        # iou = 2/6 = 1/3 = 0.3333
        assert 0.332 < iou(self.a, self.b) < 0.334

    def test_dice(self):
        # intersection = 2
        # tensor sums = 4+4 = 8
        # dice = 2 * 1/4 = 0.5
        assert dice(self.a, self.b) == 0.5

    def test_dice_loss(self):
        assert dice_loss(self.a, self.b) == 0.5


if __name__ == '__main__':
    unittest.main()
