import time
import unittest

import torch

from src.ml.model import HUNet, InceptionBlock, DownBlock, DownBlockWithPool, BotBlock, UpBlock


class TestModelForward(unittest.TestCase):

    def test_inception_block_forward(self):
        # pytorch shape convention: (n_samples, channels, height, width)
        in_channels = 6
        iblock_channels = 3
        xx = torch.randn(1, in_channels, 12, 12)

        out_channels = iblock_channels*3
        model = InceptionBlock(in_channels, iblock_channels)
        out = model(xx)
        assert out.shape == (1, out_channels, 12, 12)

    def test_down_block_forward(self):
        in_channels = 6
        iblock_channels = 3
        xx = torch.randn(1, in_channels, 12, 12)

        out_channels = iblock_channels*3 + in_channels
        model = DownBlock(in_channels, iblock_channels)
        out = model(xx)
        assert out.shape == (1, out_channels, 12, 12)

    def test_down_block_with_pool_forward(self):
        in_channels = 12
        iblock_channels = 32
        in_xy_dim = 32
        xx = torch.randn(1, in_channels, in_xy_dim, in_xy_dim)

        out_xy_dim = 16
        out_channels = 108
        model = DownBlockWithPool(in_channels, iblock_channels)
        out = model(xx)
        assert out.shape == (1, out_channels, out_xy_dim, out_xy_dim)

    def test_bot_block_forward(self):
        in_channels = 6
        iblock_channels = 3
        in_xy_dim = 12
        xx = torch.randn(1, in_channels, in_xy_dim, in_xy_dim)

        out_xy_dim = in_xy_dim//2
        out_channels = iblock_channels*3 + in_channels
        model = BotBlock(in_channels, iblock_channels, dropout=0.3)
        out = model(xx)
        assert out.shape == (1, out_channels, out_xy_dim, out_xy_dim)

    def test_up_block_forward(self):
        up_in_channels = 18
        skip_in_channels = 9
        iblock_channels = 3
        up_in_xy_dim = 6
        skip_in_xy_dim = 12

        up_xx = torch.randn(1, up_in_channels, up_in_xy_dim, up_in_xy_dim)
        skip_xx = torch.randn(1, skip_in_channels, skip_in_xy_dim, skip_in_xy_dim)

        out_xy_dim = up_in_xy_dim*2
        out_channels = iblock_channels*4 + skip_in_channels
        model = UpBlock(up_in_channels, skip_in_channels, iblock_channels)
        out = model(up_xx, skip_xx)
        assert out.shape == (1, out_channels, out_xy_dim, out_xy_dim)

    def test_hunet_forward(self):
        in_channels = 12
        in_xy_dim = 256
        in_shape = (1, in_channels, in_xy_dim, in_xy_dim)
        xx = torch.randn(in_shape)

        model = HUNet(in_channels)
        tic = time.time()
        out = model(xx)
        toc = time.time()
        print(f'\n*** HUNet forward time: {toc-tic} ***')
        assert out.shape == in_shape


if __name__ == '__main__':
    unittest.main()

