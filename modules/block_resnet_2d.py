import os
import unicodedata
import string
import glob
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import random
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import tensorboardX
import argparse
import datetime
import sklearn
import sklearn.model_selection
import logging
import json

# TODO https://github.com/D-X-Y/ResNeXt-DenseNet/blob/master/models/resnext.py
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

class BasicBlock2D(nn.Module):
    bottleneck_resize_factor = 2

    def __init__(self, in_channels, out_channels, is_conv_bias):
        super(BasicBlock2D, self).__init__()

        self.is_bottle_neck = (in_channels != out_channels)


        if not self.is_bottle_neck:
            self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                   stride=1,
                                   padding=1, bias=is_conv_bias)
            self.bn1 = nn.BatchNorm2d(in_channels)

            self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                   stride=1,
                                   padding=1, bias=is_conv_bias)
            self.bn2 = nn.BatchNorm2d(in_channels)

            self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                   stride=1,
                                   padding=1, bias=is_conv_bias)
            self.bn3 = nn.BatchNorm2d(in_channels)
        else:
            self.conv1 = nn.Conv2d(in_channels, in_channels,
                                   kernel_size=1,
                                   stride=1,
                                   padding=1, bias=is_conv_bias)
            self.bn1 = nn.BatchNorm2d(in_channels)

            self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2,
                                   padding=0, bias=is_conv_bias)
            self.bn2 = nn.BatchNorm2d(in_channels)

            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                                   padding=0, bias=is_conv_bias)
            self.bn3 = nn.BatchNorm2d(out_channels)

            # TODO identity conv
            self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2,
                                   padding=1, bias=is_conv_bias)
            self.bn_res = nn.BatchNorm2d(out_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Batch, Channel, W
        residual = x

        out = self.conv1(x)
        out = F.relu(out, inplace=True)
        out = self.bn1(out)

        out = self.conv2(out)
        out = F.relu(out, inplace=True)
        out = self.bn2(out)

        out = self.conv3(out)
        out = F.relu(out, inplace=True)
        out = self.bn3(out)

        if self.is_bottle_neck:
            residual = self.conv_res(residual)
            residual = F.relu(residual, inplace=True)
            residual = self.bn_res(residual)

        out += residual #adding
        out = F.relu(out, inplace=True)

        return out