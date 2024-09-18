from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    r"""Personnal implementation of ResBlocks, used for ResNets on Cifar-10."""

    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        stride: int = 1,
        downsample_fun: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_chan, out_chan, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.ds_fun = downsample_fun

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        block_input = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.ds_fun is not None:
            block_input = self.ds_fun(x)
        out += block_input
        out = self.relu(out)
        return out


class CifarResNet(nn.Module):
    r"""Corresponds to the implementation made for CIFAR-10 in https://doi.org/10.1109/CVPR.2016.90"""

    def __init__(self, n: int, out_dim: int, input_chans: int = 3):
        super(CifarResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_chans, 16, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.block1 = nn.Sequential(*[ResBlock(16, 16) for i in range(n)])

        self.block2 = nn.Sequential(
            ResBlock(
                16,
                32,
                stride=2,
                downsample_fun=nn.Sequential(
                    nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(32),
                ),
            ),
            *[ResBlock(32, 32) for i in range(n - 1)]
        )

        self.block3 = nn.Sequential(
            ResBlock(
                32,
                64,
                stride=2,
                downsample_fun=nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(64),
                ),
            ),
            *[ResBlock(64, 64) for i in range(n - 1)]
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.block1(out)

        out = self.block2(out)

        out = self.block3(out)

        out = self.avg_pool(out).squeeze()

        out = self.fc(out)

        return out
