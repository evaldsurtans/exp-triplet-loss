import multiprocessing
import os
import unicodedata
import string
import glob
import io
from multiprocessing import Process

import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.sampler
import random
import time
import math
import torchnet as tnt # pip install git+https://github.com/pytorch/tnt.git@master

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from joblib import Parallel, delayed
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
from numpy import dot
from numpy.linalg import norm
import logging

from modules.dict_to_obj import DictToObj
from modules.file_utils import FileUtils
from modules.math_utils import cosine_similarity, normalize_vec
from modules.metrics_utils import MetricAccuracyClassification


def get_distance(x1, x2, triplet_similarity, mode='numpy'):
    n_jobs = 8 if x1.shape[0] > 1 else 1
    is_item = False
    if isinstance(x1, np.ndarray):
        if len(x1.shape) == 1:
            is_item = True
            x1 = np.expand_dims(x1, 0)
            x2 = np.expand_dims(x2, 0)
    else:
        if len(x1.size()) == 1:
            x1 = x1.unsqueeze(0)
            x2 = x2.unsqueeze(0)
            is_item = True

    if mode != 'numpy':
        if isinstance(x1, np.ndarray):
            x1 = torch.FloatTensor(x1).to(mode)
            x2 = torch.FloatTensor(x2).to(mode)

    # latest needed: conda install -c anaconda scikit-learn
    if isinstance(x1, np.ndarray):
        with sklearn.config_context(working_memory=1024):
            if triplet_similarity == 'cos':
                dist = np.zeros((0, ))
                for each in sklearn.metrics.pairwise.pairwise_distances_chunked(x1, x2, metric="cosine", n_jobs=n_jobs):
                    dist = np.concatenate((dist, np.diag(each)), axis=0)
            else:
                dist = np.zeros((0, ))
                for each in sklearn.metrics.pairwise.pairwise_distances_chunked(x1, x2, metric="euclidean", n_jobs=n_jobs):
                    dist = np.concatenate((dist, np.diag(each)), axis=0)
    else:
        if triplet_similarity == 'cos':
            dist = 1. - F.cosine_similarity(x1, x2, dim=1, eps=1e-20) # -1 .. 1 => 0 .. 2
        else:
            dist = F.pairwise_distance(x1, x2, eps=1e-20) # 0 .. 2

    if mode != 'numpy':
        if isinstance(x1, np.ndarray):
            dist = dist.to('cpu').numpy()

    if is_item:
        dist = dist[0]

    return dist

def process_dists(idx_start, y_each, y_list, path_embeddings, sample_count, classes_size, embedding_size, triplet_similarity, mode):
    try:
        path_emb_json = f'{path_embeddings}/{y_each}.json'
        path_emb_mem = f'{path_embeddings}/{y_each}.mmap'
        path_dists_mem = f'{path_embeddings}/dists.mmap'

        dists_mem = np.memmap(
            path_dists_mem,
            mode='r+',
            dtype=np.float16,
            shape=(sample_count, classes_size))

        emb_json = FileUtils.loadJSON(path_emb_json)
        emb_mem = np.memmap(
            path_emb_mem,
            mode='r',
            dtype=np.float16,
            shape=(emb_json['count'], embedding_size))

        path_centroids_mem = f'{path_embeddings}/dists.mmap'
        centroids_mem = np.memmap(
            path_centroids_mem,
            mode='r',
            dtype=np.float16,
            shape=(classes_size, embedding_size))

        for idx_y in y_list:
            np_class_centroids_tiled = np.tile(centroids_mem[idx_y], (emb_json['count'], 1))
            dists = get_distance(emb_mem, np_class_centroids_tiled, triplet_similarity, mode).tolist()
            dists_mem[idx_start:idx_start+emb_json['count'], idx_y] = dists[:]
        #dists_mem.flush()
    except Exception as e:
        logging.error(str(e))
        exc_type, exc_value, exc_tb = sys.exc_info()
        logging.error('\n'.join(traceback.format_exception(exc_type, exc_value, exc_tb)))


# @type = ['range', 'closest']
# @mode = ['numpy', 'cuda', 'cpu']
def calculate_accuracy(
        path_embeddings,
        meter_acc: tnt.meter.ClassErrorMeter,
        meter_auc: tnt.meter.AUCMeter,
        type='range',
        norm='l2',
        triplet_similarity='cos',
        mode='cpu',
        embedding_size=None,
        class_max_dist=None, # precomputed
        class_centroids=None,
        y_list=None, #precumputed
        sample_count=None, #precomputed
        paths_embs_idx_path_pairs=None): # precomputed

    paths_embs = FileUtils.listSubFiles(path_embeddings)

    # calculate centroids first
    if class_max_dist is None:
        class_centroids = {}
        class_max_dist = {}
        y_list = []
        paths_embs_idx_path_pairs = []
        sample_count = 0

        for path_emb in paths_embs:
            if path_emb.endswith('.json'):
                y_each = int(os.path.basename(path_emb).split('.')[0])
                path_emb_json = f'{path_embeddings}/{y_each}.json'
                path_emb_mem = f'{path_embeddings}/{y_each}.mmap'

                emb_json = FileUtils.loadJSON(path_emb_json)
                emb_mem = np.memmap(
                    path_emb_mem,
                    mode='r',
                    dtype=np.float16,
                    shape=(emb_json['count'], embedding_size))

                paths_embs_idx_path_pairs.append((sample_count, y_each))
                sample_count += emb_json['count']

                y_list += (np.ones((emb_json['count'], ), dtype=np.int) * y_each).tolist()

                class_centroids[y_each] = np.average(emb_mem, axis=0)
                if norm == 'l2':
                    class_centroids[y_each] = normalize_vec(class_centroids[y_each])

                np_class_centroids_tiled = np.tile(class_centroids[y_each], (len(emb_mem), 1))
                list_dists = get_distance(np_class_centroids_tiled, emb_mem, triplet_similarity, mode).tolist()
                list_dists = sorted(list_dists, reverse=False)
                list_dists = list_dists[:max(2, int(len(list_dists) * 0.9))] # drop 10 top percent embeddings as they could contain noise
                class_max_dist[y_each] = list_dists[-1] # last largest distance

    classes_size = int(np.max(y_list)) + 1

    # store distance matrix as memmap for optimization
    path_dists_mem = f'{path_embeddings}/dists.mmap'
    is_exist_dists_mem = os.path.exists(path_dists_mem)
    dists_mem = np.memmap(
        path_dists_mem,
        mode='r+' if is_exist_dists_mem else 'w+',
        dtype=np.float16,
        shape=(sample_count, classes_size))
    #dists_mem.flush()

    path_centroids_mem = f'{path_embeddings}/dists.mmap'
    is_exist_centroids_mem = os.path.exists(path_centroids_mem)
    centroids_mem = np.memmap(
        path_centroids_mem,
        mode='r+' if is_exist_centroids_mem else 'w+',
        dtype=np.float16,
        shape=(classes_size, embedding_size))
    for key, value in class_centroids.items():
        centroids_mem[key] = value
    #centroids_mem.flush()

    if not is_exist_dists_mem:
        Parallel(n_jobs=multiprocessing.cpu_count() * 2, backend='threading')(
            delayed(process_dists)(
                idx_start, y_each, y_list, path_embeddings, sample_count, classes_size, embedding_size, triplet_similarity, mode
            )
            for idx_start, y_each in paths_embs_idx_path_pairs
        )

        dists_mem = np.memmap(
            path_dists_mem,
            mode='r',
            dtype=np.float16,
            shape=(sample_count, classes_size))


    # iterate through precomputed distances to add to data to meters for mem optimization
    chunk_size = 1024
    for idx_chunk_start in range(sample_count//chunk_size + 1):
        idx_chunk_end = min(sample_count, idx_chunk_start + chunk_size)
        chunk_each_size = idx_chunk_end - idx_chunk_start

        if chunk_each_size == 0:
            break

        if type == 'range':
            predicted = np.zeros( (chunk_each_size, classes_size), dtype=np.float)
        else:
            predicted = np.ones( (chunk_each_size, classes_size), dtype=np.float) * 1e9
        target = np.zeros( (chunk_each_size, classes_size), dtype=np.float)

        for idx_y in class_max_dist.keys():
            max_dist = class_max_dist[idx_y]
            for idx_class in range(chunk_each_size):
                target[idx_class, y_list[idx_chunk_start+idx_class]] = 1.0

            dists = dists_mem[idx_chunk_start:idx_chunk_end]

            if type == 'range':
                for idx_emb, dist in enumerate(dists):
                    if max_dist > dist[idx_y]:
                        predicted[idx_emb, idx_y] += 1.0
            else:
                predicted[:,idx_y] = np.minimum(predicted[:,idx_y], dists[:, idx_y]) # store for each class closest embedding with distance value

        if type == 'range':
            predicted = predicted / (np.sum(predicted, axis=1, keepdims=True) + 1e-18)
        else:
            # TODO softmax/hardmax based accuracy
            idx_class = np.argmin(predicted, axis=1) # for each sample select closest distance
            predicted = np.zeros_like(predicted) # init probabilities vector
            predicted[np.arange(predicted.shape[0]), idx_class] = 1.0 # for each sample set prob 100% by columns
        y_chunk = np.array(y_list[idx_chunk_start:idx_chunk_end])
        meter_acc.add(predicted, y_chunk)

        # AssertionError: targets should be binary (0, 1)
        idxes_classes = np.argmax(predicted, axis=1)
        target_tp = np.array(np.equal(y_chunk, idxes_classes), dtype=np.int)
        meter_auc.add(np.max(predicted, axis=1), target_tp)

    return class_max_dist, class_centroids, y_list, sample_count, paths_embs_idx_path_pairs


