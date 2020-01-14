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

# pre-activated relu
class ResBlock2D(nn.Module):

    def __init__(self, in_channels, out_channels, is_conv_bias, kernel_size, stride):
        super().__init__()

        kernel_size_first = kernel_size
        padding = int(kernel_size/2)
        if kernel_size % 2 == 0:
            kernel_size_first = kernel_size - 1
            padding -= 1

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size_first,
                               stride=1,
                               padding=int(kernel_size_first/2), bias=is_conv_bias)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride,
                               padding=padding, bias=is_conv_bias)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.is_projection = False
        if stride > 1 or in_channels != out_channels:
            self.is_projection = True
            self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=stride,
                                   padding=0, bias=is_conv_bias)

    def forward(self, x):
        # Batch, Channel, W

        out = self.conv1(x)
        out = F.relu(out, inplace=True)
        out = self.bn1(out)

        out = self.conv2(out)

        if self.is_projection:
            residual = self.conv_res(x)
        else:
            residual = x

        out += residual #adding
        out = F.relu(out, inplace=True)
        out = self.bn2(out)

        return out


class DeResBlock2D(nn.Module):

    def __init__(self, out_channels, in_channels, is_conv_bias, kernel_size, stride):
        super().__init__()

        kernel_size_first = kernel_size
        padding = int(kernel_size/2)
        if kernel_size % 2 == 0:
            kernel_size_first = kernel_size - 1
            padding -= 1

        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size_first,
                               stride=1,
                               padding=int(kernel_size_first/2), bias=is_conv_bias)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride,
            padding=padding, bias=is_conv_bias)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.is_projection = False
        if stride > 1 or in_channels != out_channels:
            self.is_projection = True
            self.conv_res = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=stride,
                padding=0, bias=is_conv_bias)

    def forward(self, x):
        # Batch, Channel, W

        out = self.conv1(x)
        out = F.relu(out, inplace=True)
        out = self.bn1(out)

        out = self.conv2(out)

        if self.is_projection:
            residual = self.conv_res(x)
        else:
            residual = x

        out += residual #adding
        out = F.relu(out, inplace=True)
        out = self.bn2(out)

        return out


class ConvBlock2D(nn.Module):

    def __init__(self, in_channels, out_channels, is_conv_bias, kernel_size, stride):
        super().__init__()

        padding = int(kernel_size/2)
        if kernel_size % 2 == 0:
            padding -= 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride,
                               padding=padding, bias=is_conv_bias)

        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Batch, Channel, W
        out = self.conv1(x)
        out = F.relu(out, inplace=True)
        out = self.bn1(out)
        return out


class DeConvBlock2D(nn.Module):

    def __init__(self, out_channels, in_channels, is_conv_bias, kernel_size, stride):
        super().__init__()

        padding = int(kernel_size/2)
        if kernel_size % 2 == 0:
            padding -= 1

        self.conv1 = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=is_conv_bias)

        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Batch, Channel, W
        out = self.conv1(x)
        out = F.relu(out, inplace=True)
        out = self.bn1(out)
        return out