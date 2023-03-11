import torch
from torch import nn

"""
Define U-Net model
"""


# 激活函数在神经网络中的作用有很多，主要作用是给神经网络提供非线性建模能力
# 如果没有激活函数，那么再多层的神经网络也只能处理线性可分问题
# 采用LeakyReLU激活函数；ReLU(x)=max(x,0)
# 初始定义double conv卷积层；in_channel, out_channel为相应的输入与输出
class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            # 为了保持图片scale不变，这里的kernel_size=3, padding=1, stride=1，不偏置bias
            # padding_mode:填充模式，reflect：镜像填充，最后一个像素不镜像，
            # 例如[1,2,3,4] -> [3,2,1,2,3,4,3,2]
            # （由于最后一个像素不镜像，所以跳过1和4，分别从2和3开始进行镜像填充）；
            nn.BatchNorm2d(out_channel),
            # 现在主流的都会加一个BN，并且保持scale不变
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


# 下采样模块
class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            # 使用卷积进行2倍的下采样，通道数不变
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


# 上采样模块
class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1)
        self.up = torch.nn.ConvTranspose2d(channel, channel, 2, 2)
        # torch.nn.ConvTranspose2d 转置卷积

    def forward(self, x, feature_map):
        out = self.layer(self.up(x))
        # 拼接，当前上采样的，和之前下采样过程中的
        return torch.cat((out, feature_map), dim=1)


# class UNet主干网络架构
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        # 4次下采样
        self.c1 = Conv_Block(in_channels, 64)  # in_channels=64,改小in_channel
        self.d1 = DownSample(64)
        self.c2 = Conv_Block(64, 128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(128, 256)
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(256, 512)
        self.d4 = DownSample(512)
        self.c5 = Conv_Block(512, 1024)

        # 4次上采样
        self.u1 = UpSample(1024)
        self.c6 = Conv_Block(1024, 512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.u4 = UpSample(128)
        self.c9 = Conv_Block(128, 64)
        self.out = nn.Conv2d(64, out_channels, 3, 1, 1)
        self.norml = nn.Sigmoid()
        # sigmoid函数将输入变换为(0,1)上的输出


    def forward(self, x):
        R1 = self.c1(x)
        # print(R1.size())
        # R1为最初经过conv的size
        # 定义下采样部分
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        # print(R5.size())
        # R5为网络底部的size
        # 定义上采样部分，需要和之前的拼接起来
        O1 = self.c6(self.u1(R5, R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))
        out = self.norml(self.out(O4))
        return out

# if __name__ == '__main__':
#     x = torch.randn(16, 1, 256, 256)
#     net = UNet(1, 1)
#     print("shape: ", net(x).shape)
