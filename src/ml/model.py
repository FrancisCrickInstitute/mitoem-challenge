import torch
import torch.nn as nn
import torch.nn.functional as F


def leaky_relu(x_in, negative_slope=0.1):
    return F.leaky_relu(x_in, negative_slope=negative_slope)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, iblock_channels):
        super(InceptionBlock, self).__init__()
        # path 1
        self.p1_conv2d_1x1 = nn.Conv2d(in_channels, iblock_channels, (1, 1))
        self.p1_conv2d_3x3 = nn.Conv2d(iblock_channels, iblock_channels, (3, 3), padding=1)
        # path 2
        self.p2_conv2d_1x1 = nn.Conv2d(in_channels, iblock_channels, (1, 1))
        self.p2_conv2d_5x5 = nn.Conv2d(iblock_channels, iblock_channels, (5, 5), padding=2)
        # path 3
        self.p3_max_pool2d = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.p3_conv2d_1x1 = nn.Conv2d(in_channels, iblock_channels, (1, 1))

    def forward(self, x_in: torch.Tensor):
        # path 1
        x1 = self.p1_conv2d_1x1(x_in)
        x1 = leaky_relu(x1)
        x1 = self.p1_conv2d_3x3(x1)
        x1 = leaky_relu(x1)
        # path 2
        x2 = self.p2_conv2d_1x1(x_in)
        x2 = leaky_relu(x2)
        x2 = self.p2_conv2d_5x5(x2)
        x2 = leaky_relu(x2)
        # path 3
        x3 = self.p3_max_pool2d(x_in)
        x3 = leaky_relu(x3)
        x3 = self.p3_conv2d_1x1(x3)
        x3 = leaky_relu(x3)
        # concat channels
        return torch.cat((x1, x2, x3), dim=1)


class DownBlock(nn.Module):
    def __init__(self, in_channels, iblock_channels):
        super(DownBlock, self).__init__()
        self.iblock = InceptionBlock(in_channels, iblock_channels)

    def forward(self, x_in):
        x1 = self.iblock(x_in)
        return torch.cat((x_in, x1), dim=1)


class DownBlockWithPool(nn.Module):
    def __init__(self, in_channels, iblock_channels):
        super(DownBlockWithPool, self).__init__()
        self.max_pool2d = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.iblock = InceptionBlock(in_channels, iblock_channels)

    def forward(self, x_in):
        x_pool = self.max_pool2d(x_in)
        x = self.iblock(x_pool)
        return torch.cat((x_pool, x), dim=1)


class BotBlock(nn.Module):
    def __init__(self, in_channels, iblock_channels, dropout):
        super(BotBlock, self).__init__()
        # note on batch norm:
        #     in tensorflow version: epsilon=0.001, momentum=0.99
        #     pytorch defaults: epsilon=1e-5, momentum=0.1
        self.max_pool2d = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.dropout = nn.Dropout(dropout)
        self.iblock = InceptionBlock(in_channels, iblock_channels)

    def forward(self, x_in):
        x1 = self.max_pool2d(x_in)
        x2 = self.batch_norm(x1)
        x2 = self.dropout(x2)
        x2 = self.iblock(x2)
        return torch.cat((x1, x2), dim=1)


class UpBlock(nn.Module):
    def __init__(self, up_in_channels, skip_in_channels, iblock_channels):
        super(UpBlock, self).__init__()
        iblock_in_channels = skip_in_channels+iblock_channels
        self.conv2d_transpose = nn.ConvTranspose2d(up_in_channels, iblock_channels, (3, 3), stride=2, padding=1, output_padding=1)
        self.iblock = InceptionBlock(iblock_in_channels, iblock_channels)

    def forward(self, x_in_up, x_in_skip):
        x1 = self.conv2d_transpose(x_in_up)
        x1 = leaky_relu(x1)
        x1 = torch.cat((x1, x_in_skip), dim=1)
        x2 = self.iblock(x1)
        return torch.cat((x1, x2), dim=1)


class HUNet(nn.Module):
    def __init__(self, in_channels, start_iblock_channels=32, dropout=0.3):
        super(HUNet, self).__init__()
        # 12x256x256 input - crop from image volume
        # 12 channels on front as per pytorch convention
        iblock2_channels = start_iblock_channels*2
        iblock3_channels = iblock2_channels*2
        iblock4_channels = iblock3_channels*2
        iblock5_channels = iblock4_channels*2
        block1_out_channels = in_channels + start_iblock_channels*3
        block2_out_channels = block1_out_channels + iblock2_channels*3
        block3_out_channels = block2_out_channels + iblock3_channels*3
        block4_out_channels = block3_out_channels + iblock4_channels*3
        up4_in_channels = block4_out_channels + iblock5_channels*3
        up3_in_channels = block4_out_channels + iblock4_channels*4
        up2_in_channels = block3_out_channels + iblock3_channels*4
        up1_in_channels = block2_out_channels + iblock2_channels*4
        final_conv_in_channels = block1_out_channels + start_iblock_channels*4

        self.down_block1 = DownBlock(in_channels, start_iblock_channels)
        self.down_block2 = DownBlockWithPool(block1_out_channels, iblock2_channels)
        self.down_block3 = DownBlockWithPool(block2_out_channels, iblock3_channels)
        self.down_block4 = DownBlockWithPool(block3_out_channels, iblock4_channels)

        self.bot_block = BotBlock(block4_out_channels, iblock5_channels, dropout=dropout)

        self.up_block4 = UpBlock(up4_in_channels, block4_out_channels, iblock4_channels)
        self.up_block3 = UpBlock(up3_in_channels, block3_out_channels, iblock3_channels)
        self.up_block2 = UpBlock(up2_in_channels, block2_out_channels, iblock2_channels)
        self.up_block1 = UpBlock(up1_in_channels, block1_out_channels, start_iblock_channels)

        self.final_conv2d = nn.Conv2d(final_conv_in_channels, in_channels, (1, 1))

    def forward(self, x_in):
        # encoder path
        down1 = self.down_block1(x_in)
        down2 = self.down_block2(down1)
        down3 = self.down_block3(down2)
        down4 = self.down_block4(down3)
        # bottom
        bottom = self.bot_block(down4)
        # decoder path + skip connections
        up4 = self.up_block4(bottom, down4)
        up3 = self.up_block3(up4, down3)
        up2 = self.up_block2(up3, down2)
        up1 = self.up_block1(up2, down1)

        out = self.final_conv2d(up1)
        return torch.sigmoid(out)

