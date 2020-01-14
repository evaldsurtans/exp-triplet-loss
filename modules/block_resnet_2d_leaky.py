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

    def __init__(self, in_channels, out_channels, is_conv_bias, kernel_size, stride, leaky_relu_slope):
        super(ResBlock2D, self).__init__()
        self.leaky_relu_slope = leaky_relu_slope

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                               stride=1,
                               padding=int(kernel_size/2), bias=is_conv_bias)

        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride,
                               padding=int(kernel_size/2), bias=is_conv_bias)

        self.is_projection = False
        if stride > 1 or in_channels != out_channels:
            self.is_projection = True
            self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                                   padding=0, bias=is_conv_bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Batch, Channel, W

        #print(f'cell')
        #print(x.shape)
        out = self.bn1(x)
        out = F.leaky_relu(out, negative_slope=self.leaky_relu_slope, inplace=True)
        out = self.conv1(out)
        #print(out.shape)

        out = self.bn2(out)
        out = F.leaky_relu(out, negative_slope=self.leaky_relu_slope, inplace=True)
        out = self.conv2(out)
        #print(out.shape)

        if self.is_projection:
            residual = self.conv_res(x)
        else:
            residual = x

        out += residual #adding

        return out

class ConvBlock2D(nn.Module):

    def __init__(self, in_channels, out_channels, is_conv_bias, kernel_size, stride):
        super(ConvBlock2D, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride,
                               padding=int(kernel_size/2), bias=is_conv_bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Batch, Channel, W
        return self.conv1(x)