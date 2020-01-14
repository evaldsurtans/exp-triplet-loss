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

# just zip together batch by pairs, do not mine anything

class TripletSampler(object):

    def __init__(self, args):
        self.args = args
        super(TripletSampler, self).__init__()

    def get_distance(self, x1, x2):
        if self.args.triplet_similarity == 'cos':
            dist = 1. - F.cosine_similarity(x1, x2, dim=0, eps=1e-20) # -1 .. 1 => 0 .. 2
        else:
            dist = F.pairwise_distance(x1.unsqueeze(0), x2.unsqueeze(0), eps=1e-20) # 0 .. 2
            if self.args.triplet_similarity == 'euclidean_2':
                dist = dist ** 2
        return dist

    def sample_batch(self, output, y, margin, max_distance=2.0):
        positives_dist = []
        negatives_dist = []

        for idx in range(y.size(0)):
            if idx % self.args.triplet_positives == 0:
                for idx_pos in range(idx + 1, idx + self.args.triplet_positives):
                    positives_dist.append(self.get_distance(output[idx], output[idx_pos]))

            idx_to_neg = self.args.triplet_positives - idx % self.args.triplet_positives
            for idx_neg in range(idx + idx_to_neg, y.size(0)):
                dist = self.get_distance(output[idx], output[idx_neg])
                if not torch.equal(y[idx_neg], y[idx]): # in case repeated samples in batch
                    negatives_dist.append(dist)
                else:
                    positives_dist.append(dist)

        positives_dist = torch.stack(positives_dist)
        negatives_dist = torch.stack(negatives_dist)
        result = dict(
            positives_dist = positives_dist,
            negatives_dist = negatives_dist,
            positives_dist_all_filtred = positives_dist,
            negatives_dist_all_filtred = negatives_dist,
            positives_dist_all = positives_dist,
            negatives_dist_all = negatives_dist,
        )

        return result
