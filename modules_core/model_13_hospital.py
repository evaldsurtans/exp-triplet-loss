import os
import unicodedata
import string
import glob
import io
import torch
import torch.nn
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

import torch
import torch.nn as nn
import torchvision.models
import torchvision.models.resnet
import torch.utils.model_zoo

from modules.block_pass import Pass
from modules.block_resnet_2d_leaky import ResBlock2D, ConvBlock2D
from modules.block_reshape2 import Reshape
from modules.dict_to_obj import DictToObj
import modules.torch_utils as torch_utils
from modules.layer_kaf import KAF, KAF2D
from modules.layer_maxout import MaxoutLinear

# Variant without affline, fully convolutional net
# fixed refined part pooling

class Model(torch.nn.Module):
    def __init__(self, args):

        super(Model, self).__init__()
        self.args = args

        if not hasattr(self.args, 'embedding_function'):
            self.args.embedding_function = 'tanh'

        layers_encoder, output_size = self.__create_layers_encoder(name='enc', input_size=args.input_size)

        logging.info(f'conv outputs: {output_size}x{output_size} {self.channels_conv_size} ')

        self.layers_encoder = layers_encoder

        if args.embedding_layers > 0:
            input_size_emb = self.channels_conv_size * output_size**2
            self.layers_embedding = torch.nn.Sequential(
                Reshape(shape=input_size_emb)
            )
            for idx in range(args.embedding_layers):
                output_size_emb = args.embedding_layers_hidden
                if idx == args.embedding_layers - 1:
                    output_size_emb = args.embedding_size

                if self.args.layers_embedding_dropout > 0:
                    self.layers_embedding.add_module(f'dropout_{idx}', torch.nn.Dropout(p=self.args.layers_embedding_dropout))

                self.layers_embedding.add_module(f'emb_linear_{idx}', torch.nn.Linear(input_size_emb, output_size_emb, bias=False))
                if idx < args.embedding_layers - 1:
                    self.layers_embedding.add_module(f'emb_relu_linear_{idx}', torch.nn.LeakyReLU(negative_slope=self.args.leaky_relu_slope))
                input_size_emb = output_size_emb
        elif args.layers_embedding_type == 'pooled':
            self.layers_embedding = torch.nn.Sequential(
                Reshape(shape=(1, self.channels_conv_size))
            )

            if self.args.layers_embedding_dropout > 0:
                self.layers_embedding.add_module(f'dropout_emb', torch.nn.Dropout(p=self.args.layers_embedding_dropout))

            self.layers_embedding.add_module(f'pool_emb', torch.nn.AdaptiveAvgPool1d(output_size=self.args.embedding_size))
            self.layers_embedding.add_module(f'last_reshape', Reshape(shape=self.args.embedding_size))
        else:
            func = torch.nn.ReLU()

            if args.layers_embedding_type == 'refined':

                if self.args.suffix_affine_layers_hidden_func == 'kaf':
                    func = KAF(num_parameters=self.channels_conv_size, D=self.args.suffix_affine_layers_hidden_params)
                elif self.args.suffix_affine_layers_hidden_func == 'maxout':
                    func = MaxoutLinear(1, 1, pool_size=self.args.suffix_affine_layers_hidden_params)

                func_dropout = Pass()
                if self.args.layers_embedding_dropout > 0:
                    func_dropout = torch.nn.Dropout(p=self.args.layers_embedding_dropout)

                # refined part pooling version
                # https://arxiv.org/pdf/1711.09349.pdf
                self.layers_embedding = torch.nn.Sequential(
                    torch.nn.BatchNorm2d(num_features=self.channels_conv_size),
                    func,
                    func_dropout,
                    torch.nn.Conv2d(kernel_size=1, stride=1, in_channels=self.channels_conv_size, out_channels=self.args.embedding_size, bias=False),
                    Reshape(shape=self.args.embedding_size)
                )
            elif args.layers_embedding_type == 'last': # last
                self.layers_embedding = torch.nn.Sequential(
                    Reshape(shape=self.channels_conv_size),
                    torch.nn.BatchNorm1d(num_features=self.channels_conv_size)
                )

                if self.args.layers_embedding_dropout > 0:
                    self.layers_embedding.add_module(f'dropout_emb', torch.nn.Dropout(p=self.args.layers_embedding_dropout))

                if self.args.suffix_affine_layers_hidden_func == 'kaf':
                    self.layers_embedding.add_module('emb_last_lin', torch.nn.Linear(in_features=self.channels_conv_size, out_features=self.args.embedding_size))
                    self.layers_embedding.add_module('emb_last', KAF(num_parameters=self.args.embedding_size, D=self.args.suffix_affine_layers_hidden_params))
                elif self.args.suffix_affine_layers_hidden_func == 'maxout':
                    self.layers_embedding.add_module('emb_last', MaxoutLinear(d_in=self.channels_conv_size, d_out=self.args.embedding_size, pool_size=self.args.suffix_affine_layers_hidden_params))
                else:
                    self.layers_embedding.add_module('emb_last_lin', torch.nn.Linear(in_features=self.channels_conv_size, out_features=self.args.embedding_size))

        if self.args.embedding_function == 'sigmoid':
            self.layers_embedding.add_module('emb_sigmoid', torch.nn.Sigmoid())
        elif self.args.embedding_function == 'tanh':
            self.layers_embedding.add_module('emb_tanh', torch.nn.Tanh())
        elif self.args.embedding_function == 'hard_tanh':
            self.layers_embedding.add_module('emb_hard_tanh', torch.nn.Hardtanh())
        elif self.args.embedding_function == 'hard_sink':
            self.layers_embedding.add_module('emb_hard_sink', torch.nn.Hardshrink())
        elif self.args.embedding_function == 'tanh_shrink':
            self.layers_embedding.add_module('emb_tanh_shrink', torch.nn.Tanhshrink())

        if self.args.embedding_layers_last_norm != 'none':
            self.layers_embedding.add_module('emb_last_reshape', Reshape(shape=(1, self.args.embedding_size)))

        if self.args.embedding_layers_last_norm == 'instance':
            self.layers_embedding.add_module('emb_last_norm', torch.nn.InstanceNorm1d(1))
        elif self.args.embedding_layers_last_norm == 'batch':
            self.layers_embedding.add_module('emb_last_norm', torch.nn.BatchNorm1d(1))
        elif self.args.embedding_layers_last_norm == 'layer':
            self.layers_embedding.add_module('emb_last_norm', torch.nn.LayerNorm(1))
        elif self.args.embedding_layers_last_norm == 'local':
            self.layers_embedding.add_module('emb_last_norm', torch.nn.LocalResponseNorm(1))

        if self.args.embedding_layers_last_norm != 'none':
            self.layers_embedding.add_module('emb_last_reshape_final', Reshape(shape=self.args.embedding_size))

        if self.args.datasource_classes_train:
            logging.info(f'classification layer classes count: {self.args.datasource_classes_train}')
            layers_classification = []
            layers_emb_size = self.args.embedding_size

            for idx in range(self.args.class_layers - 1):
                layers_classification += [
                    torch.nn.BatchNorm1d(num_features=layers_emb_size),
                    torch.nn.Linear(in_features=layers_emb_size, out_features=self.args.class_layers_hidden)
                ]
                layers_emb_size = self.args.class_layers_hidden

            layers_classification += [
                torch.nn.BatchNorm1d(num_features=layers_emb_size),
                torch.nn.Linear(in_features=layers_emb_size, out_features=self.args.datasource_classes_train),
                torch.nn.Softmax(dim=1)
            ]

            self.layers_classification = torch.nn.Sequential(
                *layers_classification
            )

        torch_utils.init_parameters(self.layers_embedding)

        for name, param in self.layers_embedding.named_parameters():
            if param.requires_grad:
                if name.startswith('emb_') and 'bias' not in name:
                    if len(param.size()) > 1:
                        if args.embedding_init == 'xavier':
                            torch.nn.init.xavier_uniform_(param)
                        elif args.embedding_init == 'xavier_normal':
                            torch.nn.init.xavier_normal_(param)
                        elif args.embedding_init == 'uniform':
                            torch.nn.init.uniform_(param)
                        elif args.embedding_init == 'normal':
                            torch.nn.init.normal_(param)
                        elif args.embedding_init == 'zeros' or args.embedding_init == 'zero':
                            torch.nn.init.zeros_(param)
                        elif args.embedding_init == 'ones' or args.embedding_init == 'one':
                            torch.nn.init.ones_(param)


    def __create_layers_encoder(self, name, input_size):

        self.channels_conv_size = 1 #inout channels

        if self.args.model_encoder == 'resnet18':
            model_pretrained = torchvision.models.resnet18(pretrained=True)
        elif self.args.model_encoder == 'resnet34':
            model_pretrained = torchvision.models.resnet34(pretrained=True)
        elif self.args.model_encoder == 'resnet50':
            model_pretrained = torchvision.models.resnet50(pretrained=True)
        elif self.args.model_encoder == 'resnet101':
            model_pretrained = torchvision.models.resnet101(pretrained=True)
        elif self.args.model_encoder == 'densenet121':
            model_pretrained = torchvision.models.densenet121(pretrained=True)
        elif self.args.model_encoder == 'densenet161':
            model_pretrained = torchvision.models.densenet161(pretrained=True)

        if self.args.model_encoder.startswith('resnet'):

            if self.args.input_features != model_pretrained.conv1.in_channels:
                weight_conv1_pretrained = model_pretrained.conv1.weight.data
                model_pretrained.conv1 = torch.nn.Conv2d(self.args.input_features, model_pretrained.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)

                idx_rgb = 0
                for idx in range(self.args.input_features):
                    model_pretrained.conv1.weight.data[:, idx, :, :] = weight_conv1_pretrained[:, idx_rgb, :, :]
                    idx_rgb += 1
                    if idx_rgb == 2:
                        idx_rgb = 0

            modules = list(model_pretrained.children())[:-1] # drop FC
            layers_conv = torch.nn.Sequential(*modules)

            input_size = 1
            self.channels_conv_size = model_pretrained.fc.in_features
        elif self.args.model_encoder.startswith('densenet'):
            conv1 = list(model_pretrained.features.children())[0]
            if self.args.input_features != conv1.in_channels:
                weight_conv1_pretrained = conv1.weight.data
                conv1 = torch.nn.Conv2d(self.args.input_features, conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)

                idx_rgb = 0
                for idx in range(self.args.input_features):
                    conv1.weight.data[:, idx, :, :] = weight_conv1_pretrained[:, idx_rgb, :, :]
                    idx_rgb += 1
                    if idx_rgb == 2:
                        idx_rgb = 0

                is_first = True
                features = torch.nn.Sequential()
                for name, module in model_pretrained.features.named_children():
                    if is_first:
                        module = conv1
                        is_first = False
                    features.add_module(name, module)
                model_pretrained.features = features

            modules = list(model_pretrained.children())[:-1] # drop FC
            modules.append(torch.nn.AdaptiveAvgPool2d(output_size=1))
            layers_conv = torch.nn.Sequential(*modules)

            input_size = 1
            self.channels_conv_size = model_pretrained.classifier.in_features

            if not self.args.is_model_encoder_pretrained:
                torch_utils.init_parameters(layers_conv)

        return layers_conv, input_size

    def forward(self, x):
        # debug code
        # inp = x
        # for name, each in self.layers_encoder.named_children():
        #     print(f'{name} in: {inp.size()}')
        #     out = each.forward(inp)
        #     print(f'{name} out: {out.size()}')
        #     inp = out

        output_enc = self.layers_encoder.forward(x)
        output_emb = self.layers_embedding.forward(output_enc)

        if self.args.embedding_function != 'none':
            output_emb = output_emb * self.args.embedding_scale

        output_norm = torch_utils.normalize_output(
            output_emb,
            self.args.embedding_norm,
            self.args.embedding_scale)

        output_class = None
        if self.training:
            output_class = self.layers_classification.forward(output_norm)
        return output_norm, output_class
