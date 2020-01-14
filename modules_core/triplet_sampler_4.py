import os
import unicodedata
import string
import glob
import io
import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.sampler
import random
import time
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import tensorboardX
import argparse
from datetime import datetime
import sklearn
import sklearn.model_selection
from enum import Enum
import json
import logging
import numpy as np
import traceback, sys
import itertools

from modules.dict_to_obj import DictToObj

class TripletSampler(object):

    def __init__(self, args):
        self.args = args
        super(TripletSampler, self).__init__()

    def get_distances(self, output):
        # cosine distance http://mlwiki.org/index.php/Cosine_Similarity

        # d == 0 when exactly same, d == 1 when different
        # !! CANNOT HAVE too big batches ﻿64×64 = 4096

        # go through all batch
        distances_batch = []
        batch_size = output.size(0)
        for i in range(batch_size):

            x1 = output[i, :].expand(batch_size, -1)
            x2 = output # whole batch

            if self.args.triplet_similarity == 'cos':
                distances = 1. - F.cosine_similarity(x1, x2, dim=1, eps=1e-20) # -1 .. 1 => 0 .. 2
            else:
                distances = F.pairwise_distance(x1, x2, eps=1e-20) # 0 .. 2

            distances_batch.append(distances)

        return torch.stack(distances_batch)

    def sample_batch(self, output, y, margin, max_distance=2.0):

        positves_pairs = []
        negatives_pairs = []
        anchors = []

        positives_dist = []
        negatives_dist = []

        positives_dist_all = []
        negatives_dist_all = []

        positives_dist_all_filtred = []
        negatives_dist_all_filtred = []

        distances_batch = self.get_distances(output)

        step = 1
        if self.args.triplet_sampler_var == 'hard':
            step = self.args.triplet_positives

        for idx_anchor in range(0, y.size(0), step): #one negative per positives
            anchor = output[idx_anchor]
            anchor_y = y[idx_anchor]

            anchor_distances = distances_batch[idx_anchor]

            positive = None
            positive_dist = None
            positive_y = None
            for idx_positive in range(y.size(0)):
                each_y = y[idx_positive]
                if idx_positive != idx_anchor and torch.equal(each_y, anchor_y): # same class
                    if idx_positive > idx_anchor:
                        positives_dist_all.append(anchor_distances[idx_positive])

                    if 'abs_margin' in self.args.filter_samples or 'abs_margin_asym' in self.args.filter_samples:
                        if anchor_distances[idx_positive]/max_distance < margin: # if under margin then skip (already in positive range)
                            continue

                    if idx_positive > idx_anchor:
                        positives_dist_all_filtred.append(anchor_distances[idx_positive])

                    if positive is None:
                        positive_y = each_y
                        positive = output[idx_positive]
                        positive_dist = anchor_distances[idx_positive]
                    elif positive_dist < anchor_distances[idx_positive]: # find worst case positive where longest distance
                        positive_y = each_y
                        positive_dist = anchor_distances[idx_positive]
                        positive = output[idx_positive]

            negative = None
            negative_dist = None
            negative_y = None
            for idx_negative in range(y.size(0)): # must iterate all for negatives_dist_all and repeated samples in batch size
                each_y = y[idx_negative]
                if idx_negative != idx_anchor and not torch.equal(each_y, anchor_y): # not same class
                    if idx_negative > idx_anchor:
                        # n = n_same_class_triplet * n_classes! / ((n_classes - 2)! * 2!)
                        # 3×10!/((10−2)!×2!) when batch 30
                        # 114/3 = 38 => 3×38!/((38−2)!×2!) = 2109
                        negatives_dist_all.append(anchor_distances[idx_negative])

                    if 'abs_margin' in self.args.filter_samples or 'abs_margin_asym' in self.args.filter_samples:
                        if anchor_distances[idx_negative]/max_distance > 0.5:  # if over half then skip (already in futher away as needed)
                            continue

                    # this line will execute only after getting positive_dist and then will apply to pair of samples
                    if 'hard' in self.args.filter_samples or 'semi_hard' in self.args.filter_samples:
                        if positive_dist + margin <= anchor_distances[idx_negative]:
                            #print('violate hard')
                            continue # skip violated pair

                    # this line will execute only after getting positive_dist and then will apply to pair of samples
                    if 'semi_hard' in self.args.filter_samples:
                        if anchor_distances[idx_negative] <= positive_dist:
                            #print('violate semi')
                            continue # skip violated pair

                    if idx_negative > idx_anchor:
                        negatives_dist_all_filtred.append(anchor_distances[idx_negative])
                    if negative is None:
                        negative_y = each_y
                        negative = output[idx_negative]
                        negative_dist = anchor_distances[idx_negative]
                    elif negative_dist > anchor_distances[idx_negative]:  # find worst case negative where shortest distance
                        negative_y = each_y
                        negative_dist = anchor_distances[idx_negative]
                        negative = output[idx_negative]

            if 'abs_margin_asym' in self.args.filter_samples or 'exp' in self.args.triplet_loss:
                if negative is not None:
                    negatives_dist.append(negative_dist)
                if positive is not None:
                    positives_dist.append(positive_dist)
            else:
                if negative is not None and positive is not None:
                    negatives_dist.append(negative_dist)
                    positives_dist.append(positive_dist)

            anchors.append(anchor)

            if positive is not None:
                positves_pairs.append((anchor_y, positive_y)) # must be from same y class
            if negative is not None:
                negatives_pairs.append((anchor_y, negative_y)) # must be from different y classes

        loss_shape_zeros = 0
        if 'abs_margin_asym' in self.args.filter_samples:
            loss_shape_zeros = 1

        if len(positives_dist) == 0:
            positives_dist = torch.zeros((loss_shape_zeros,)).to(self.args.device)
        else:
            positives_dist = torch.stack(positives_dist)

        if len(positives_dist_all_filtred) == 0:
            positives_dist_all_filtred = torch.zeros((loss_shape_zeros,)).to(self.args.device)
        else:
            positives_dist_all_filtred = torch.stack(positives_dist_all_filtred)

        if len(negatives_dist) == 0:
            negatives_dist = torch.zeros((loss_shape_zeros,)).to(self.args.device)
        else:
            negatives_dist = torch.stack(negatives_dist)

        if len(negatives_dist_all_filtred) == 0:
            negatives_dist_all_filtred = torch.zeros((loss_shape_zeros,)).to(self.args.device)
        else:
            negatives_dist_all_filtred = torch.stack(negatives_dist_all_filtred)

        result = dict(
            positives_dist = positives_dist,
            negatives_dist = negatives_dist,
            positives_dist_all_filtred = positives_dist_all_filtred,
            negatives_dist_all_filtred = negatives_dist_all_filtred,
            positives_dist_all = torch.stack(positives_dist_all),
            negatives_dist_all = torch.stack(negatives_dist_all),
        )

        return result