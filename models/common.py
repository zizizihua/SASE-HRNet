# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import json
import math
import platform
import warnings
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.cuda import amp
import torch.utils.checkpoint as cp

from utils.datasets import exif_transpose, letterbox
from utils.general import (LOGGER, check_requirements, check_suffix, check_version, colorstr, increment_path,
                           make_divisible, non_max_suppression, scale_coords, xywh2xyxy, xyxy2xywh, channel_shuffle)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import copy_attr, time_sync
from utils.activations import SiLU

if not hasattr(nn, 'SiLU'):
    nn.SiLU = SiLU


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class DepthSepConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, act=True):
        super().__init__()
        self.dw_conv = Conv(c1, c1, k=k, s=s, g=c1, act=False)
        self.pw_conv = Conv(c1, c2, k=1, act=act)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


# mobile squeeze-expand block
class MSEBlock(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 ch_ratio=8,
                 stride=1,
                 kernel_size=5,
                 act=True,
                 shortcut=True):
        super().__init__()
        self.dw_conv = Conv(
            in_channels,
            in_channels,
            k=kernel_size,
            s=stride,
            g=in_channels,
            act=act)

        mid_channels = in_channels // ch_ratio
        self.pw_conv1 = Conv(
            in_channels, 
            mid_channels, 
            k=1,
            act=False)

        self.pw_conv2 = Conv(
            mid_channels, 
            out_channels, 
            k=1,
            act=act)

        self.shortcut = shortcut and (out_channels == in_channels)
    
    def forward(self, x):
        identity = x
        x = self.dw_conv(x)
        x = self.pw_conv1(x)
        x = self.pw_conv2(x)

        return identity + x if self.shortcut else x


class InvertedBottleneck(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 ch_ratio=8,
                 stride=1,
                 kernel_size=3,
                 act=nn.ReLU(inplace=True),
                 shortcut=True):
        super().__init__()
        mid_channels = in_channels * ch_ratio
        self.pw_conv1 = Conv(
            in_channels, 
            mid_channels, 
            k=1,
            act=act)
        
        self.dw_conv = Conv(
            mid_channels,
            mid_channels,
            k=kernel_size,
            s=stride,
            g=in_channels,
            act=act)

        self.pw_conv2 = Conv(
            mid_channels, 
            out_channels, 
            k=1,
            act=False)

        self.shortcut = shortcut and (out_channels == in_channels)
    
    def forward(self, x):
        identity = x
        x = self.pw_conv1(x)
        x = self.dw_conv(x)
        x = self.pw_conv2(x)

        return identity + x if self.shortcut else x


class SpatialWeighting(nn.Module):
    def __init__(self,
                 channels,
                 ratio=16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            channels,
            int(channels / ratio),
            kernel_size=1,
            bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            int(channels / ratio),
            channels,
            kernel_size=1,
            bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.relu(self.conv1(out))
        out = self.sigmoid(self.conv2(out))
        return x * out


class CrossResolutionWeighting(nn.Module):
    def __init__(self,
                 channels,
                 ratio=16):
        super().__init__()
        self.channels = channels
        total_channel = sum(channels)
        self.conv1 = Conv(
            total_channel,
            int(total_channel / ratio),
            k=1,
            act=nn.ReLU())
        self.conv2 = Conv(
            int(total_channel / ratio),
            total_channel,
            k=1,
            act=nn.Sigmoid())

    def forward(self, x):
        mini_size = x[-1].size()[-2:]
        out = [F.adaptive_avg_pool2d(s, mini_size) for s in x[:-1]] + [x[-1]]
        out = torch.cat(out, dim=1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = torch.split(out, self.channels, dim=1)
        out = [
            s * F.interpolate(a, size=s.size()[-2:], mode='nearest')
            for s, a in zip(x, out)
        ]
        return out


class ConditionalChannelWeighting(nn.Module):

    def __init__(self,
                 in_channels,
                 stride,
                 reduce_ratio,
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.stride = stride
        assert stride in [1, 2]

        branch_channels = [channel // 2 for channel in in_channels]

        self.cross_resolution_weighting = CrossResolutionWeighting(
            branch_channels,
            ratio=reduce_ratio)

        self.depthwise_convs = nn.ModuleList([
            Conv(
                channel,
                channel,
                k=3,
                s=self.stride,
                p=1,
                g=channel,
                act=None) for channel in branch_channels
        ])

        self.spatial_weighting = nn.ModuleList([
            SpatialWeighting(channels=channel, ratio=4)
            for channel in branch_channels
        ])

    def forward(self, x):

        def _inner_forward(x):
            x = [s.chunk(2, dim=1) for s in x]
            x1 = [s[0] for s in x]
            x2 = [s[1] for s in x]

            x2 = self.cross_resolution_weighting(x2)
            x2 = [dw(s) for s, dw in zip(x2, self.depthwise_convs)]
            x2 = [sw(s) for s, sw in zip(x2, self.spatial_weighting)]

            out = [torch.cat([s1, s2], dim=1) for s1, s2 in zip(x1, x2)]
            out = [channel_shuffle(s, 2) for s in out]

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out

class LiteHRStem(nn.Module):

    def __init__(self,
                 in_channels,
                 stem_channels,
                 out_channels,
                 expand_ratio=1,
                 with_cp=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.with_cp = with_cp

        self.conv1 = Conv(
            in_channels,
            stem_channels,
            k=3,
            s=2,
            p=1,
            act=nn.ReLU(inplace=True))

        mid_channels = int(round(stem_channels * expand_ratio))
        branch_channels = stem_channels // 2
        if stem_channels == self.out_channels:
            inc_channels = self.out_channels - branch_channels
        else:
            inc_channels = self.out_channels - stem_channels

        self.branch1 = nn.Sequential(
            Conv(
                branch_channels,
                branch_channels,
                k=3,
                s=2,
                p=1,
                g=branch_channels,
                act=None),
            Conv(
                branch_channels,
                inc_channels,
                k=1,
                act=nn.ReLU(inplace=True)),
        )

        self.expand_conv = Conv(
            branch_channels,
            mid_channels,
            k=1,
            act=nn.ReLU(inplace=True))
        self.depthwise_conv = Conv(
            mid_channels,
            mid_channels,
            k=3,
            s=2,
            p=1,
            g=mid_channels,
            act=None)
        self.linear_conv = Conv(
            mid_channels,
            branch_channels
            if stem_channels == self.out_channels else stem_channels,
            k=1,
            act=nn.ReLU(inplace=True))

    def forward(self, x):

        def _inner_forward(x):
            x = self.conv1(x)
            x1, x2 = x.chunk(2, dim=1)

            x2 = self.expand_conv(x2)
            x2 = self.depthwise_conv(x2)
            x2 = self.linear_conv(x2)

            out = torch.cat((self.branch1(x1), x2), dim=1)

            out = channel_shuffle(out, 2)

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


class ShuffleUnit(nn.Module):
    """InvertedResidual block for ShuffleNetV2 backbone.
    Args:
        in_channels (int): The input channels of the block.
        out_channels (int): The output channels of the block.
        stride (int): Stride of the 3x3 convolution layer. Default: 1
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 act=nn.ReLU(inplace=True),
                 with_cp=False):
        super().__init__()
        self.stride = stride
        self.with_cp = with_cp

        branch_features = out_channels // 2
        # if self.stride == 1:
        #     assert in_channels == branch_features * 2, (
        #         f'in_channels ({in_channels}) should equal to '
        #         f'branch_features * 2 ({branch_features * 2}) '
        #         'when stride is 1')

        # if in_channels != branch_features * 2:
        #     assert self.stride != 1, (
        #         f'stride ({self.stride}) should not equal 1 when '
        #         f'in_channels != branch_features * 2')

        if self.stride > 1 or in_channels != branch_features * 2:
            self.branch1 = nn.Sequential(
                Conv(
                    in_channels,
                    in_channels,
                    k=3,
                    s=self.stride,
                    p=1,
                    g=in_channels,
                    act=None),
                Conv(
                    in_channels,
                    branch_features,
                    k=1,
                    act=act),
            )
        else:
            self.branch1 = None

        self.branch2 = nn.Sequential(
            Conv(
                in_channels if self.branch1 is not None else branch_features,
                branch_features,
                k=1,
                act=act),
            Conv(
                branch_features,
                branch_features,
                k=3,
                s=self.stride,
                p=1,
                g=branch_features,
                act=None),
            Conv(
                branch_features,
                branch_features,
                k=1,
                act=act))

    def forward(self, x):

        def _inner_forward(x):
            if self.branch1 is not None :
                out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
            else:
                x1, x2 = x.chunk(2, dim=1)
                out = torch.cat((x1, self.branch2(x2)), dim=1)

            out = channel_shuffle(out, 2)

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


class LiteHRModule(nn.Module):

    def __init__(
            self,
            num_branches,
            num_blocks,
            in_channels,
            reduce_ratio,
            module_type,
            fuse_method='add',
            kernel_size=3,
            multiscale_output=True,
            with_fuse=True,
            with_cp=False,
    ):
        super().__init__()
        self._check_branches(num_branches, in_channels)

        self.in_channels = in_channels
        self.num_branches = num_branches

        self.module_type = module_type
        self.multiscale_output = multiscale_output
        self.with_fuse = with_fuse
        self.fuse_method = fuse_method
        self.with_cp = with_cp

        if self.module_type == 'LITE':
            self.layers = self._make_weighting_blocks(num_blocks, reduce_ratio)
        elif self.module_type == 'NAIVE':
            self.layers = self._make_naive_branches(num_branches, num_blocks)
        elif self.module_type == 'MSE':
            self.layers = self._make_mse_branches(num_branches, num_blocks, reduce_ratio, kernel_size)
        elif self.module_type == 'IBN':
            self.layers = self._make_ibn_branches(num_branches, num_blocks, reduce_ratio)
        if self.with_fuse:
            self._make_fuse_layers()
            self.relu = nn.ReLU()

    def _check_branches(self, num_branches, in_channels):
        """Check input to avoid ValueError."""
        if num_branches != len(in_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                f'!= NUM_INCHANNELS({len(in_channels)})'
            raise ValueError(error_msg)

    def _make_weighting_blocks(self, num_blocks, reduce_ratio, stride=1):
        layers = []
        for i in range(num_blocks):
            layers.append(
                ConditionalChannelWeighting(
                    self.in_channels,
                    stride=stride,
                    reduce_ratio=reduce_ratio,
                    with_cp=self.with_cp))

        return nn.Sequential(*layers)
    
    def _make_one_branch_mse(self, branch_index, num_blocks, reduce_ratio, kernel_size=5, stride=1):
        """Make one branch."""
        layers = []
        layers.append(
            MSEBlock(
                self.in_channels[branch_index],
                self.in_channels[branch_index],
                ch_ratio=reduce_ratio,
                kernel_size=kernel_size,
                stride=stride))
        for i in range(1, num_blocks):
            layers.append(
                MSEBlock(
                    self.in_channels[branch_index],
                    self.in_channels[branch_index],
                    ch_ratio=reduce_ratio,
                    kernel_size=kernel_size,
                    stride=1))
        
        return nn.Sequential(*layers)
    
    def _make_one_branch_ibn(self, branch_index, num_blocks, reduce_ratio, stride=1):
        """Make one branch."""
        layers = []
        layers.append(
            InvertedBottleneck(
                self.in_channels[branch_index],
                self.in_channels[branch_index],
                ch_ratio=reduce_ratio,
                stride=stride))
        for i in range(1, num_blocks):
            layers.append(
                InvertedBottleneck(
                    self.in_channels[branch_index],
                    self.in_channels[branch_index],
                    ch_ratio=reduce_ratio,
                    stride=1))
        
        return nn.Sequential(*layers)

    def _make_one_branch(self, branch_index, num_blocks, stride=1):
        """Make one branch."""
        layers = []
        layers.append(
            ShuffleUnit(
                self.in_channels[branch_index],
                self.in_channels[branch_index],
                stride=stride,
                with_cp=self.with_cp))
        for i in range(1, num_blocks):
            layers.append(
                ShuffleUnit(
                    self.in_channels[branch_index],
                    self.in_channels[branch_index],
                    stride=1,
                    with_cp=self.with_cp))

        return nn.Sequential(*layers)

    def _make_naive_branches(self, num_branches, num_blocks):
        """Make branches."""
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch(i, num_blocks))

        return nn.ModuleList(branches)
    
    def _make_mse_branches(self, num_branches, num_blocks, reduce_ratio, kernel_size):
        """Make branches."""
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch_mse(i, num_blocks, reduce_ratio, kernel_size))

        return nn.ModuleList(branches)
    
    def _make_ibn_branches(self, num_branches, num_blocks, reduce_ratio):
        """Make branches."""
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch_ibn(i, num_blocks, reduce_ratio))

        return nn.ModuleList(branches)
    
    def _make_fuse_layers(self):
        if self.fuse_method == 'add':
            self._make_add_fuse_layers()
            self.forward = self._add_fuse_forward
        elif self.fuse_method == 'attn':
            self._make_attn_fuse_layers()
            self.forward = self._attn_fuse_forward
        else:
            raise NotImplementedError

    def _make_add_fuse_layers(self):
        """Make fuse layer."""
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []

        for i in range(num_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            Conv(
                                in_channels[j],
                                in_channels[i],
                                k=1,
                                act=False),
                            nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                DepthSepConv(
                                    in_channels[j],
                                    in_channels[i],
                                    k=3,
                                    s=2,
                                    act=False))

                        else:
                            conv_downsamples.append(
                                DepthSepConv(
                                    in_channels[j],
                                    in_channels[j],
                                    k=3,
                                    s=2,
                                    act=False))

                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        self.fuse_layers = nn.ModuleList(fuse_layers)
    
    def _make_attn_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels

        self.upsamples = nn.ModuleList()
        for i in range(1, num_branches):
            self.upsamples.append(nn.Sequential(
                nn.Upsample(scale_factor=2),
                Conv(
                    in_channels[i],
                    in_channels[i-1],
                    k=1)))

        self.upconvs = nn.ModuleList() if num_branches > 2 else []
        for i in range(1, num_branches - 1):
            self.upconvs.append(
                DepthSepConv(
                    in_channels[i],
                    in_channels[i],
                    k=3))

        self.downsamples = nn.ModuleList()
        for i in range(num_branches - 1):
            self.downsamples.append(
                DepthSepConv(
                    in_channels[i],
                    in_channels[i + 1],
                    k=3,
                    s=2))

        self.downconvs = nn.ModuleList()
        for i in range(num_branches):
            self.downconvs.append(
                DepthSepConv(
                    in_channels[i],
                    in_channels[i],
                    k=3))

        size = 2 * (num_branches - 2)
        self.fpn_up_weights = nn.Parameter(
            torch.ones(size), requires_grad=True) if size > 0 else None
        size = 2 + 3 * (num_branches - 2) + 2
        self.fpn_down_weights = nn.Parameter(
            torch.ones(size), requires_grad=True)

    def _add_fuse_forward(self, x):
        if self.module_type == 'LITE':
            x = self.layers(x)
        else:
            for i in range(self.num_branches):
                x[i] = self.layers[i](x[i])

        if self.with_fuse:
            x_fuse = []
            for i in range(len(self.fuse_layers)):
                y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
                for j in range(self.num_branches):
                    if i == j:
                        y += x[j]
                    else:
                        y += self.fuse_layers[i][j](x[j])
                x_fuse.append(F.relu(y))
            x = x_fuse

        return x
    
    def _attn_fuse_forward(self, x):
        if self.module_type == 'LITE':
            x = self.layers(x)
        else:
            for i in range(self.num_branches):
                x[i] = self.layers[i](x[i])

        if self.with_fuse:
            y = [None] * self.num_branches
            y[self.num_branches-1] = x[self.num_branches-1]
            # up
            for i in range(self.num_branches - 1, 0, -1):
                if i == 1:
                    y[i-1] = self.upsamples[i-1](y[i])
                else:
                    w = F.relu(self.fpn_up_weights[(i - 2) * 2:(i - 1) * 2])
                    w = w / w.sum()
                    y[i-1] = self.upconvs[i-2](w[0] * self.upsamples[i-1](y[i]) + w[1] * x[i-1])

            # down
            for i in range(self.num_branches):
                if i == 0:
                    w = F.relu(self.fpn_down_weights[:2])
                    w = w / w.sum()
                    x[i] = self.downconvs[i](w[0] * y[i] + w[1] * x[i])
                elif i == self.num_branches - 1:
                    w = F.relu(self.fpn_down_weights[-2:])
                    w = w / w.sum()
                    x[i] = self.downconvs[i](
                        w[0] * self.downsamples[i-1](x[i-1]) + w[1] * x[i])
                else:
                    w = F.relu(
                        self.fpn_down_weights[2 + (i - 1) * 3:2 + i * 3])
                    w = w / w.sum()
                    x[i] = self.downconvs[i](
                        w[0] * self.downsamples[i-1](x[i-1]) + w[1] * x[i] + w[2] * y[i])

        return x

    # def _make_fuse_layers(self):
    #     """Make fuse layer."""
    #     if self.num_branches == 1:
    #         return None

    #     num_branches = self.num_branches
    #     in_channels = self.in_channels
    #     fuse_layers = []
    #     num_out_branches = num_branches if self.multiscale_output else 1
    #     for i in range(num_out_branches):
    #         fuse_layer = []
    #         for j in range(num_branches):
    #             if j > i:
    #                 fuse_layer.append(
    #                     nn.Sequential(
    #                         nn.Conv2d(
    #                             in_channels[j],
    #                             in_channels[i],
    #                             kernel_size=1,
    #                             stride=1,
    #                             padding=0,
    #                             bias=False),
    #                         nn.BatchNorm2d(in_channels[i]),
    #                         nn.Upsample(
    #                             scale_factor=2**(j - i), mode='nearest')))
    #             elif j == i:
    #                 fuse_layer.append(None)
    #             else:
    #                 conv_downsamples = []
    #                 for k in range(i - j):
    #                     if k == i - j - 1:
    #                         conv_downsamples.append(
    #                             nn.Sequential(
    #                                 nn.Conv2d(
    #                                     in_channels[j],
    #                                     in_channels[j],
    #                                     kernel_size=3,
    #                                     stride=2,
    #                                     padding=1,
    #                                     groups=in_channels[j],
    #                                     bias=False),
    #                                 nn.BatchNorm2d(in_channels[j]),
    #                                 nn.Conv2d(
    #                                     in_channels[j],
    #                                     in_channels[i],
    #                                     kernel_size=1,
    #                                     stride=1,
    #                                     padding=0,
    #                                     bias=False),
    #                                 nn.BatchNorm2d(in_channels[i])))
    #                     else:
    #                         conv_downsamples.append(
    #                             nn.Sequential(
    #                                 nn.Conv2d(
    #                                     in_channels[j],
    #                                     in_channels[j],
    #                                     kernel_size=3,
    #                                     stride=2,
    #                                     padding=1,
    #                                     groups=in_channels[j],
    #                                     bias=False),
    #                                 nn.BatchNorm2d(in_channels[j]),
    #                                 nn.Conv2d(
    #                                     in_channels[j],
    #                                     in_channels[j],
    #                                     kernel_size=1,
    #                                     stride=1,
    #                                     padding=0,
    #                                     bias=False),
    #                                 nn.BatchNorm2d(in_channels[j]),
    #                                 nn.ReLU(inplace=True)))
    #                 fuse_layer.append(nn.Sequential(*conv_downsamples))
    #         fuse_layers.append(nn.ModuleList(fuse_layer))

    #     return nn.ModuleList(fuse_layers)

    # def forward(self, x):
    #     """Forward function."""
    #     if self.num_branches == 1:
    #         return [self.layers[0](x[0])]

    #     if self.module_type == 'LITE':
    #         out = self.layers(x)
    #     else:
    #         for i in range(self.num_branches):
    #             x[i] = self.layers[i](x[i])
    #         out = x

    #     if self.with_fuse:
    #         out_fuse = []
    #         for i in range(len(self.fuse_layers)):
    #             y = out[0] if i == 0 else self.fuse_layers[i][0](out[0])
    #             for j in range(self.num_branches):
    #                 if i == j:
    #                     y += out[j]
    #                 else:
    #                     y += self.fuse_layers[i][j](out[j])
    #             out_fuse.append(self.relu(y))
    #         out = out_fuse
    #     elif not self.multiscale_output:
    #         out = [out[0]]
    #     return out


class SASEModule(nn.Module):
    def __init__(
            self,
            in_channels,
            num_blocks,
            ch_ratio,
            fuse_method='attn',
            kernel_size=5):
        super().__init__()
        self.in_channels = in_channels
        self.num_branches = len(in_channels)
        self.ch_ratio = ch_ratio
        self.kernel_size = kernel_size

        self.with_fuse = fuse_method != 'none'
        self.fuse_method = fuse_method

        self.layers = self._make_branches(num_blocks)
        if self.with_fuse:
            self._make_fuse_layers()

    def _make_one_branch(self, in_channels, num_blocks):
        return nn.Sequential(*[
            MSEBlock(
                in_channels,
                in_channels,
                ch_ratio=self.ch_ratio,
                kernel_size=self.kernel_size)
            for _ in range(num_blocks)])

    def _make_branches(self, num_blocks):
        """Make branches."""
        layers = nn.ModuleList()
        for ch in self.in_channels:
            layers.append(self._make_one_branch(ch, num_blocks))

        return layers

    def _make_fuse_layers(self):
        if self.fuse_method == 'add':
            self._make_add_fuse_layers()
            self.forward = self._add_fuse_forward
        elif self.fuse_method == 'attn':
            self._make_attn_fuse_layers()
            self.forward = self._attn_fuse_forward
        else:
            raise NotImplementedError

    def _make_add_fuse_layers(self):
        """Make fuse layer."""
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []

        for i in range(num_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            Conv(
                                in_channels[j],
                                in_channels[i],
                                k=1,
                                act=False),
                            nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                DepthSepConv(
                                    in_channels[j],
                                    in_channels[i],
                                    k=3,
                                    s=2,
                                    act=False))

                        else:
                            conv_downsamples.append(
                                DepthSepConv(
                                    in_channels[j],
                                    in_channels[j],
                                    k=3,
                                    s=2,
                                    act=False))

                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        self.fuse_layers = nn.ModuleList(fuse_layers)
    
    def _make_attn_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels

        self.upsamples = nn.ModuleList()
        for i in range(1, num_branches):
            self.upsamples.append(nn.Sequential(
                nn.Upsample(scale_factor=2),
                Conv(
                    in_channels[i],
                    in_channels[i-1],
                    k=1)))

        self.upconvs = nn.ModuleList() if num_branches > 2 else []
        for i in range(1, num_branches - 1):
            self.upconvs.append(
                DepthSepConv(
                    in_channels[i],
                    in_channels[i],
                    k=3))

        self.downsamples = nn.ModuleList()
        for i in range(num_branches - 1):
            self.downsamples.append(
                DepthSepConv(
                    in_channels[i],
                    in_channels[i + 1],
                    k=3,
                    s=2))

        self.downconvs = nn.ModuleList()
        for i in range(num_branches):
            self.downconvs.append(
                DepthSepConv(
                    in_channels[i],
                    in_channels[i],
                    k=3))

        size = 2 * (num_branches - 2)
        self.fpn_up_weights = nn.Parameter(
            torch.ones(size), requires_grad=True) if size > 0 else None
        size = 2 + 3 * (num_branches - 2) + 2
        self.fpn_down_weights = nn.Parameter(
            torch.ones(size), requires_grad=True)

    def _add_fuse_forward(self, x):
        for i in range(self.num_branches):
            x[i] = self.layers[i](x[i])

        if self.with_fuse:
            x_fuse = []
            for i in range(len(self.fuse_layers)):
                y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
                for j in range(self.num_branches):
                    if i == j:
                        y += x[j]
                    else:
                        y += self.fuse_layers[i][j](x[j])
                x_fuse.append(F.relu(y))
            x = x_fuse

        return x
    
    def _attn_fuse_forward(self, x):
        for i in range(self.num_branches):
            x[i] = self.layers[i](x[i])
        
        if self.with_fuse:
            y = [None] * self.num_branches
            y[self.num_branches-1] = x[self.num_branches-1]
            # up
            for i in range(self.num_branches - 1, 0, -1):
                if i == 1:
                    y[i-1] = self.upsamples[i-1](y[i])
                else:
                    w = F.relu(self.fpn_up_weights[(i - 2) * 2:(i - 1) * 2])
                    w = w / w.sum()
                    y[i-1] = self.upconvs[i-2](w[0] * self.upsamples[i-1](y[i]) + w[1] * x[i-1])

            # down
            for i in range(self.num_branches):
                if i == 0:
                    w = F.relu(self.fpn_down_weights[:2])
                    w = w / w.sum()
                    x[i] = self.downconvs[i](w[0] * y[i] + w[1] * x[i])
                elif i == self.num_branches - 1:
                    w = F.relu(self.fpn_down_weights[-2:])
                    w = w / w.sum()
                    x[i] = self.downconvs[i](
                        w[0] * self.downsamples[i-1](x[i-1]) + w[1] * x[i])
                else:
                    w = F.relu(
                        self.fpn_down_weights[2 + (i - 1) * 3:2 + i * 3])
                    w = w / w.sum()
                    x[i] = self.downconvs[i](
                        w[0] * self.downsamples[i-1](x[i-1]) + w[1] * x[i] + w[2] * y[i])

        return x


class HRTransitionLayer(nn.Module):
    def __init__(self, c1, c2, act=nn.ReLU()):
        super().__init__()
        self.downsample = DepthSepConv(c1[-1], c2[-1], k=3, s=2, act=act)
    
    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        x.append(self.downsample(x[-1]))
        return x


class HROutputLayer(nn.Module):
    def __init__(self, c1, c2, n, out_inds, block='MSE', ratio=8, kernel_size=5):
        super().__init__()
        self.out_inds = out_inds
        if n > 0:
            if block == 'MSE':
                self.layers = self.get_mse_layers(c1, c2, n, ratio, out_inds, kernel_size)
            elif block == 'SHB':
                self.layers = self.get_shb_layers(c1, c2, n, out_inds)
            elif block == 'IBN':
                self.layers = self.get_ibn_layers(c1, c2, n, ratio, out_inds, kernel_size)
            else:
                raise NotImplementedError(block)
        else:
            self.layers = None
    
    def get_mse_layers(self, c1, c2, n, ratio, out_inds, kernel_size):
        layers = nn.ModuleList()
        for i, j in enumerate(out_inds):
            convs = []
            for _ in range(n):
                convs.append(
                    MSEBlock(c1[j], c2[i], ch_ratio=ratio, kernel_size=kernel_size, shortcut=False))
                c1[j] = c2[i]
            layers.append(nn.Sequential(*convs))
        
        return layers
    
    def get_shb_layers(self, c1, c2, n, out_inds):
        layers = nn.ModuleList()
        for i, j in enumerate(out_inds):
            convs = []
            for _ in range(n):
                convs.append(
                    ShuffleUnit(c1[j], c2[i]))
                c1[j] = c2[i]
            layers.append(nn.Sequential(*convs))
        
        return layers
    
    def get_ibn_layers(self, c1, c2, n, ratio, out_inds, kernel_size):
        layers = nn.ModuleList()
        for i, j in enumerate(out_inds):
            convs = []
            for _ in range(n):
                convs.append(
                    InvertedBottleneck(c1[j], c2[i], ch_ratio=ratio, kernel_size=kernel_size, shortcut=False))
                c1[j] = c2[i]
            layers.append(nn.Sequential(*convs))
        
        return layers
    
    def forward(self, x):
        x = [x[i] for i in self.out_inds]
        if self.layers is not None:
            x = [conv(xi) for xi, conv in zip(x, self.layers)]
        return x


class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False):
        # Usage:
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx with --dnn
        #   OpenVINO:                       *.xml
        #   CoreML:                         *.mlmodel
        #   TensorRT:                       *.engine
        #   TensorFlow SavedModel:          *_saved_model
        #   TensorFlow GraphDef:            *.pb
        #   TensorFlow Lite:                *.tflite
        #   TensorFlow Edge TPU:            *_edgetpu.tflite
        from models.experimental import attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs = self.model_type(w)  # get backend
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        #w = attempt_download(w)  # download if not local
        fp16 &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16
        if data:  # data.yaml path (optional)
            with open(data, errors='ignore') as f:
                names = yaml.safe_load(f)['names']  # class names

        if pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, map_location=device)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif jit:  # TorchScript
            LOGGER.info(f'Loading {w} for TorchScript inference...')
            extra_files = {'config.txt': ''}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files)
            model.half() if fp16 else model.float()
            if extra_files['config.txt']:
                d = json.loads(extra_files['config.txt'])  # extra_files dict
                stride, names = int(d['stride']), d['names']
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f'Loading {w} for ONNX OpenCV DNN inference...')
            check_requirements(('opencv-python>=4.5.4',))
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            cuda = torch.cuda.is_available()
            check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(w, providers=providers)
        elif xml:  # OpenVINO
            LOGGER.info(f'Loading {w} for OpenVINO inference...')
            check_requirements(('openvino-dev',))  # requires openvino-dev: https://pypi.org/project/openvino-dev/
            import openvino.inference_engine as ie
            core = ie.IECore()
            if not Path(w).is_file():  # if not *.xml
                w = next(Path(w).glob('*.xml'))  # get *.xml file from *_openvino_model dir
            network = core.read_network(model=w, weights=Path(w).with_suffix('.bin'))  # *.xml, *.bin paths
            executable_network = core.load_network(network, device_name='CPU', num_requests=1)
        elif engine:  # TensorRT
            LOGGER.info(f'Loading {w} for TensorRT inference...')
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
            check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            bindings = OrderedDict()
            fp16 = False  # default updated below
            for index in range(model.num_bindings):
                name = model.get_binding_name(index)
                dtype = trt.nptype(model.get_binding_dtype(index))
                shape = tuple(model.get_binding_shape(index))
                data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
                bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
                if model.binding_is_input(index) and dtype == np.float16:
                    fp16 = True
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            context = model.create_execution_context()
            batch_size = bindings['images'].shape[0]
        elif coreml:  # CoreML
            LOGGER.info(f'Loading {w} for CoreML inference...')
            import coremltools as ct
            model = ct.models.MLModel(w)
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            if saved_model:  # SavedModel
                LOGGER.info(f'Loading {w} for TensorFlow SavedModel inference...')
                import tensorflow as tf
                keras = False  # assume TF1 saved_model
                model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
            elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
                LOGGER.info(f'Loading {w} for TensorFlow GraphDef inference...')
                import tensorflow as tf

                def wrap_frozen_graph(gd, inputs, outputs):
                    x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                    ge = x.graph.as_graph_element
                    return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

                gd = tf.Graph().as_graph_def()  # graph_def
                gd.ParseFromString(open(w, 'rb').read())
                frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs="Identity:0")
            elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
                try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                    from tflite_runtime.interpreter import Interpreter, load_delegate
                except ImportError:
                    import tensorflow as tf
                    Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,
                if edgetpu:  # Edge TPU https://coral.ai/software/#edgetpu-runtime
                    LOGGER.info(f'Loading {w} for TensorFlow Lite Edge TPU inference...')
                    delegate = {'Linux': 'libedgetpu.so.1',
                                'Darwin': 'libedgetpu.1.dylib',
                                'Windows': 'edgetpu.dll'}[platform.system()]
                    interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
                else:  # Lite
                    LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
                    interpreter = Interpreter(model_path=w)  # load TFLite model
                interpreter.allocate_tensors()  # allocate
                input_details = interpreter.get_input_details()  # inputs
                output_details = interpreter.get_output_details()  # outputs
            elif tfjs:
                raise Exception('ERROR: YOLOv5 TF.js inference is not supported')
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, val=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.pt or self.jit:  # PyTorch
            y = self.model(im) if self.jit else self.model(im, augment=augment, visualize=visualize)
            return y if val else y[0]
        elif self.dnn:  # ONNX OpenCV DNN
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            y = self.net.forward()
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im})[0]
        elif self.xml:  # OpenVINO
            im = im.cpu().numpy()  # FP32
            desc = self.ie.TensorDesc(precision='FP32', dims=im.shape, layout='NCHW')  # Tensor Description
            request = self.executable_network.requests[0]  # inference request
            request.set_blob(blob_name='images', blob=self.ie.Blob(desc, im))  # name=next(iter(request.input_blobs))
            request.infer()
            y = request.output_blobs['output'].buffer  # name=next(iter(request.output_blobs))
        elif self.engine:  # TensorRT
            assert im.shape == self.bindings['images'].shape, (im.shape, self.bindings['images'].shape)
            self.binding_addrs['images'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = self.bindings['output'].data
        elif self.coreml:  # CoreML
            im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
            im = Image.fromarray((im[0] * 255).astype('uint8'))
            # im = im.resize((192, 320), Image.ANTIALIAS)
            y = self.model.predict({'image': im})  # coordinates are xywh normalized
            if 'confidence' in y:
                box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
                conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float)
                y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            else:
                k = 'var_' + str(sorted(int(k.replace('var_', '')) for k in y)[-1])  # output key
                y = y[k]  # output
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
            if self.saved_model:  # SavedModel
                y = (self.model(im, training=False) if self.keras else self.model(im)).numpy()
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im)).numpy()
            else:  # Lite or Edge TPU
                input, output = self.input_details[0], self.output_details[0]
                int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input['quantization']
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input['index'], im)
                self.interpreter.invoke()
                y = self.interpreter.get_tensor(output['index'])
                if int8:
                    scale, zero_point = output['quantization']
                    y = (y.astype(np.float32) - zero_point) * scale  # re-scale
            y[..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, device=self.device)
        return (y, []) if val else y

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        if any((self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb)):  # warmup types
            if self.device.type != 'cpu':  # only warmup GPU models
                im = torch.zeros(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
                for _ in range(2 if self.jit else 1):  #
                    self.forward(im)  # warmup

    @staticmethod
    def model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        from export import export_formats
        suffixes = list(export_formats().Suffix) + ['.xml']  # export suffixes
        check_suffix(p, suffixes)  # checks
        p = Path(p).name  # eliminate trailing separators
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, xml2 = (s in p for s in suffixes)
        xml |= xml2  # *_openvino_model or *.xml
        tflite &= not edgetpu  # *.tflite
        return pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs


class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    agnostic = False  # NMS class-agnostic
    multi_label = False  # NMS multiple labels per box
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    amp = False  # Automatic Mixed Precision (AMP) inference

    def __init__(self, model):
        super().__init__()
        LOGGER.info('Adding AutoShape... ')
        copy_attr(self, model, include=('yaml', 'nc', 'hyp', 'names', 'stride', 'abc'), exclude=())  # copy attributes
        self.dmb = isinstance(model, DetectMultiBackend)  # DetectMultiBackend() instance
        self.pt = not self.dmb or model.pt  # PyTorch model
        self.model = model.eval()

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_sync()]
        p = next(self.model.parameters()) if self.pt else torch.zeros(1)  # for device and type
        autocast = self.amp and (p.device.type != 'cpu')  # Automatic Mixed Precision (AMP) inference
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=autocast):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, (str, Path)):  # filename or uri
                im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                im = np.asarray(exif_transpose(im))
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, self.stride) if self.pt else size for x in np.array(shape1).max(0)]  # inf shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32
        t.append(time_sync())

        with amp.autocast(enabled=autocast):
            # Inference
            y = self.model(x, augment, profile)  # forward
            t.append(time_sync())

            # Post-process
            y = non_max_suppression(y if self.dmb else y[0], self.conf, iou_thres=self.iou, classes=self.classes,
                                    agnostic=self.agnostic, multi_label=self.multi_label, max_det=self.max_det)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_sync())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, imgs, pred, files, times=(0, 0, 0, 0), names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.times = times  # profiling times
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        crops = []
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            s = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({'box': box, 'conf': conf, 'cls': cls, 'label': label,
                                          'im': save_one_box(box, im, file=file, save=save)})
                        else:  # all others
                            annotator.box_label(box, label, color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                LOGGER.info(s.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.imgs[i] = np.asarray(im)
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    def print(self):
        self.display(pprint=True)  # print results
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' %
                    self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True) if save else None
        return self.display(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        r = range(self.n)  # iterable
        x = [Detections([self.imgs[i]], [self.pred[i]], [self.files[i]], self.times, self.names, self.s) for i in r]
        # for d in x:
        #    for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
        #        setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n

