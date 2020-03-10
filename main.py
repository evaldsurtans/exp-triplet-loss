import multiprocessing
import threading
from itertools import chain

import cv2
import matplotlib
from PIL import Image
from torch.autograd import Variable

from modules.file_utils import FileUtils
from modules.math_utils import normalize_vec
from modules.radam import RAdam
from modules.torch_utils import to_numpy

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import torch
import torch.nn
import torch.utils.data
import tensorboardX
import numpy as np
import time
import argparse
from datetime import datetime
import torch.nn.functional as F
import json
import unicodedata
import string
import glob
import io
import random
import math
import matplotlib.ticker as ticker
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import copy
import sklearn
import sklearn.model_selection
import shutil
import json
import logging

from pathlib import Path
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from skimage.transform import resize
from sklearn.metrics import roc_curve

from modules.tensorboard_utils import TensorBoardUtils
from modules.logging_utils import LoggingUtils
from modules.data_utils import DataUtils
from modules.args_utils import ArgsUtils
from modules.csv_utils import CsvUtils
from modules.dict_to_obj import DictToObj
from modules.command_txt_utils import CommandTxtUtils
from modules.mathplotlib_utils import MathPlotLibUtils
from modules.centroid_classification_utils import calculate_accuracy, get_distance

from joblib import Parallel, delayed

import torchnet as tnt # pip install git+https://github.com/pytorch/tnt.git@master
import traceback, sys

parser = argparse.ArgumentParser(description='Model trainer')

parser.add_argument('-id', default=0, type=int)
parser.add_argument('-repeat_id', default=0, type=int)
parser.add_argument('-report', default='report', type=str)
parser.add_argument('-params_report', nargs='*', required=False) # extra params for report global for series of runs
parser.add_argument('-params_report_local', nargs='*', required=False) # extra params for local run report
parser.add_argument('-name', help='Run name, by default date', default='', type=str)

parser.add_argument('-is_datasource_only', default=False, type=lambda x: (str(x).lower() == 'true')) # just test datasource
parser.add_argument('-device', default='cuda', type=str)

parser.add_argument('-model', default='model_12_dobe', type=str)
parser.add_argument('-model_encoder', default='resnet18', type=str)
parser.add_argument('-is_model_encoder_pretrained', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-layers_embedding_type', default='last', type=str) #refined, pooled, last
parser.add_argument('-layers_embedding_dropout', default=0.0, type=float)
parser.add_argument('-is_layers_embedding_batchnorm', default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument('-path_tmp_dir', default='./tmp', type=str)

parser.add_argument('-pre_trained_model', default='./tasks/test_dec29_enc_123_123.json', type=str)
parser.add_argument('-is_pretrained_locked', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-unet_preloaded_pooling_size', default=1, type=int)

parser.add_argument('-datasource', default='datasource_pytorch', type=str)
parser.add_argument('-datasource_is_grayscale', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-datasource_classes_train', default=0, type=int)
parser.add_argument('-datasource_max_class_count', default=0, type=int)

parser.add_argument('-is_class_loss', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-class_loss_coef', default=1.0, type=float)
parser.add_argument('-class_loss_epochs_limit', default=0, type=float) # 0 unlimited
parser.add_argument('-class_layers', default=1, type=int)
parser.add_argument('-class_layers_hidden', default=512, type=int)

parser.add_argument('-triplet_sampler', default='triplet_sampler_4', type=str)
parser.add_argument('-triplet_sampler_var', default='hard', type=str) # hard, all
parser.add_argument('-filter_samples', nargs='*', default=['none']) # abs_margin semi_hard hard
parser.add_argument('-triplet_similarity', default='cos', type=str) # cos euclidean euclidean_2

parser.add_argument('-embedding_norm', default='unit_range', type=str) #unit_range l2 none
parser.add_argument('-embedding_scale', default=1.0, type=float) #l2-softmax

parser.add_argument('-path_data', default='./data', type=str)
parser.add_argument('-datasource_workers', default=8, type=int) #8
parser.add_argument('-datasource_type', default='mnist', type=str) # fassion_mnist mnist, vggface2
parser.add_argument('-datasource_path_memmaps', default='/Users/evalds/Downloads/vggface2', type=str)
parser.add_argument('-datasource_exclude_train_class_ids', nargs='*', default=[])
parser.add_argument('-datasource_include_test_class_ids', nargs='*', default=[])
parser.add_argument('-datasource_size_samples', default=0, type=int) # 0 automatic use whole dataset

parser.add_argument('-epochs_count', default=20, type=int)

parser.add_argument('-optimizer', default='adam', type=str)
parser.add_argument('-learning_rate', default=1e-5, type=float)
parser.add_argument('-learning_rate_min', default=0, type=float)
parser.add_argument('-weight_decay', default=0, type=float)
parser.add_argument('-batch_size', default=114, type=int)

parser.add_argument('-triplet_positives', default=3, type=int) # ensures batch will have 2 or 3 positives (for speaker_triplet_sampler_hard must have 3)
parser.add_argument('-triplet_loss', default='exp13', type=str)
parser.add_argument('-coef_loss_neg', default=1.0, type=float)
parser.add_argument('-triplet_loss_margin', default=0.2, type=float)
parser.add_argument('-is_triplet_loss_margin_auto', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-lossless_beta', default=1.2, type=float)

parser.add_argument('-exp_coef', default=2.0, type=float)
parser.add_argument('-overlap_coef', default=1.2, type=float)

parser.add_argument('-abs_coef', default=1.5, type=float)
parser.add_argument('-tan_coef', default=20.0, type=float)
parser.add_argument('-sin_coef', default=20.0, type=float)

parser.add_argument('-slope_coef', default=3.0, type=float)
parser.add_argument('-neg_coef', default=2.0, type=float)
parser.add_argument('-pos_coef', default=3.0, type=float)

parser.add_argument('-neg_loss_coef', default=1.0, type=float)
parser.add_argument('-pos_loss_coef', default=1.0, type=float)

parser.add_argument('-noise_training', default=0.0, type=float)

parser.add_argument('-is_center_loss', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-center_loss_min_count', default=500, type=int)
parser.add_argument('-center_loss_coef', default=1.0, type=float)

parser.add_argument('-embedding_function', default='tanh', type=str)
parser.add_argument('-embedding_size', default=256, type=int)
parser.add_argument('-embedding_init', default='xavier', type=str)

parser.add_argument('-embedding_layers', default=0, type=int)
parser.add_argument('-embedding_layers_hidden', default=512, type=int)
parser.add_argument('-embedding_layers_hidden_func', default='maxout', type=str)
parser.add_argument('-embedding_layers_last_norm', default='none', type=str) # none instance batch layer local

parser.add_argument('-suffix_affine_layers', default=2, type=int)
parser.add_argument('-suffix_affine_layers_hidden', default=1024, type=int)

parser.add_argument('-conv_resnet_layers', default=3, type=int)
parser.add_argument('-conv_resnet_sub_layers', default=3, type=int)

parser.add_argument('-is_pre_grad_locked', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-pre_type', default='resnet34', type=str)

parser.add_argument('-is_conv_bias', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-conv_first_channel_count', default=32, type=int) #kvass
parser.add_argument('-conv_first_kernel', default=9, type=int)
parser.add_argument('-conv_kernel', default=3, type=int)
parser.add_argument('-conv_stride', default=2, type=int)
parser.add_argument('-conv_expansion_rate', default=2, type=float) #kvass
parser.add_argument('-is_conv_max_pool', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-is_linear_at_end', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-leaky_relu_slope', default=0.1, type=float)

parser.add_argument('-conv_unet', default='unet_add', type=str) # none, unet_add, unet_cat

parser.add_argument('-suffix_affine_layers_hidden_func', default='relu', type=str) #kaf maxout relu lin
parser.add_argument('-suffix_affine_layers_hidden_params', default=4, type=int)

parser.add_argument('-early_stopping_patience', default=5, type=int)
parser.add_argument('-early_stopping_param', default='test_dist_delta', type=str)
parser.add_argument('-early_stopping_delta_percent', default=1e-3, type=float)
parser.add_argument('-early_stopping_param_coef', default=1.0, type=float)

parser.add_argument('-is_reshuffle_after_epoch', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-is_quick_test', default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument('-max_embeddings_histograms', default=1000, type=int)
parser.add_argument('-max_embeddings_class_for_projector', default=50, type=int)
parser.add_argument('-img_size_embeddings_class_for_projector', default=32, type=int)
parser.add_argument('-max_embeddings_per_class_test', default=500, type=int) # 0 = unlimited
parser.add_argument('-max_embeddings_per_class_train', default=500, type=int) # 0 = unlimited

parser.add_argument('-max_embeddings_projector_classes', default=10, type=int) # 0 = unlimited
parser.add_argument('-max_embeddings_projector_samples', default=20, type=int) # 0 = unlimited

args, args_other = parser.parse_known_args()

# fail safe center loss
if args.max_embeddings_per_class_train > 0:
    args.center_loss_min_count = min(args.max_embeddings_per_class_train, args.center_loss_min_count)

tmp = [
    'id',
    'name',
    'repeat_id',
    'epoch',
    'test_acc_range',
    'test_acc_closest',
    'best_acc_range',
    'best_acc_closest',
    'test_eer',
    'train_acc_range',
    'train_acc_closest',
    'train_eer',
    'test_dist_delta',
    'test_dist_positives',
    'test_dist_negatives',
    'train_dist_delta',
    'train_dist_positives',
    'train_dist_negatives',
    'test_dist_delta_hard',
    'test_dist_positives_hard',
    'test_dist_negatives_hard',
    'train_dist_delta_hard',
    'train_dist_positives_hard',
    'train_dist_negatives_hard',
    'test_count_positives',
    'test_count_negatives',
    'train_count_positives',
    'train_count_negatives',
    'test_count_positives_all',
    'test_count_negatives_all',
    'train_count_positives_all',
    'train_count_negatives_all',
    'test_negative_max',
    'train_negative_max',
    'test_max_dist',
    'train_max_dist',
    'test_loss',
    'train_loss',
    'train_loss_emb',
    'train_loss_class',
    'train_loss_pos',
    'train_loss_neg',
    'test_loss_pos',
    'test_loss_neg',
    'train_loss_center',
    'test_loss_center',
    'test_count_embeddings',
    'train_count_embeddings',
    'learning_rate_dyn',
    'avg_epoch_time']
if not args.params_report is None:
    for it in args.params_report:
        if not it in tmp:
            tmp.append(it)
args.params_report = tmp

tmp = [
    'epoch',
    'test_acc_range',
    'test_acc_closest',
    'best_acc_range',
    'best_acc_closest',
    'test_eer',
    'train_acc_range',
    'train_acc_closest',
    'train_eer',
    'test_dist_delta',
    'test_dist_positives',
    'test_dist_negatives',
    'train_dist_delta',
    'train_dist_positives',
    'train_dist_negatives',
    'test_dist_delta_hard',
    'test_dist_positives_hard',
    'test_dist_negatives_hard',
    'train_dist_delta_hard',
    'train_dist_positives_hard',
    'train_dist_negatives_hard',
    'test_count_positives',
    'test_count_negatives',
    'train_count_positives',
    'train_count_negatives',
    'test_count_positives_all',
    'test_count_negatives_all',
    'train_count_positives_all',
    'train_count_negatives_all',
    'test_negative_max',
    'train_negative_max',
    'test_max_dist',
    'train_max_dist',
    'test_loss',
    'train_loss',
    'train_loss_emb',
    'train_loss_pos',
    'train_loss_neg',
    'test_loss_pos',
    'test_loss_neg',
    'train_loss_class',
    'train_loss_center',
    'test_loss_center',
    'learning_rate_dyn',
    'test_count_embeddings',
    'train_count_embeddings',
    'epoch_time',
    'early_percent_improvement']
if not args.params_report_local is None:
    for it in args.params_report_local:
        if not it in tmp:
            tmp.append(it)
args.params_report_local = tmp


if len(args.name) == 0:
    args.name = datetime.now().strftime('%y-%m-%d_%H-%M-%S')

FileUtils.createDir('./tasks/' + args.report)
run_path = './tasks/' + args.report + '/runs/' + args.name
if os.path.exists(run_path):
    shutil.rmtree(run_path, ignore_errors=True)
    time.sleep(3)
    while os.path.exists(run_path):
        time.sleep(1)

tensorboard_writer = tensorboardX.SummaryWriter(run_path)
tensorboard_utils = TensorBoardUtils(tensorboard_writer)
logging_utils = LoggingUtils(filename=os.path.join(run_path, 'log.txt'))
is_logged_cnorm = False

get_data_loaders = getattr(__import__('modules_core.' + args.datasource, fromlist=['get_data_loaders']), 'get_data_loaders')
data_loader_train, data_loader_test = get_data_loaders(args)

ArgsUtils.log_args(args, 'main.py', logging_utils)

if args.is_datasource_only:
    logging.info('is_datasource_only laoded')
    exit()

if not torch.cuda.is_available():
    args.device = 'cpu'
    logging.info('CUDA NOT AVAILABLE')
else:
    logging.info('cuda devices: {}'.format(torch.cuda.device_count()))

if __name__ == '__main__':

    Model = getattr(__import__('modules_core.' + args.model, fromlist=['Model']), 'Model')
    model = Model(args)

    TripletSampler = getattr(__import__('modules_core.' + args.triplet_sampler, fromlist=['TripletSampler']), 'TripletSampler')
    triplet_sampler = TripletSampler(args)

    # save model description (important for testing)
    with open(os.path.join(run_path + f'/{args.name}.json'), 'w') as outfile:
        json.dump(args.__dict__, outfile, indent=4)

    is_data_parallel = False
    if args.device == 'cuda' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, dim=0)
        is_data_parallel = True
        logging.info(f'PARALLEL MODEL {torch.cuda.device_count()}')

    model = model.to(args.device)

    model_module = model
    if is_data_parallel:
        model_module = model.module

    state = {
        'epoch': 0,
        'learning_rate_dyn': args.learning_rate,
        'best_param': -1,
        'avg_epoch_time': -1,
        'epoch_time': -1,
        'early_stopping_patience': 0,
        'early_percent_improvement': 0,
        'train_loss': -1,
        'test_loss': -1,
        'train_loss_class': -1,
        'test_loss_class': -1,
        'train_loss_emb': -1,
        'test_loss_emb': -1,
        'train_loss_center': -1,
        'test_loss_center': -1,
        'train_loss_pos': -1,
        'test_loss_pos': -1,
        'train_loss_neg': -1,
        'test_loss_neg': -1,
        'test_acc_range': -1,
        'best_acc_range': -1,
        'train_acc_range': -1,
        'test_auc_range': -1,
        'train_auc_range': -1,
        'test_acc_closest': -1,
        'best_acc_closest': -1,
        'train_acc_closest': -1,
        'test_auc_closest': -1,
        'train_auc_closest': -1,
        'test_max_dist': -1,
        'train_max_dist': -1,
        'test_eer': -1,
        'train_eer': -1,
        'test_eer2': -1,
        'train_eer2': -1,
        'train_dist_positives': -1,
        'train_dist_negatives': -1,
        'test_dist_positives': -1,
        'test_dist_negatives': -1,
        'test_dist_delta': -1,
        'train_dist_delta': -1,
        'train_dist_positives_hard': -1,
        'train_dist_negatives_hard': -1,
        'test_dist_positives_hard': -1,
        'test_dist_negatives_hard': -1,
        'test_dist_delta_hard': -1,
        'train_dist_delta_hard': -1,
        'train_count_positives': -1,
        'train_count_negatives': -1,
        'test_count_positives': -1,
        'test_count_negatives': -1,
        'train_count_positives_all': -1,
        'train_count_negatives_all': -1,
        'test_count_positives_all': -1,
        'test_count_negatives_all': -1,
        'test_negative_max': -1,
        'train_negative_max': -1,
        'test_count_embeddings': -1,
        'train_count_embeddings': -1
    }
    avg_time_epochs = []
    time_epoch = time.time()


    def get_optimizer(lr):
        optimizer = None
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=args.weight_decay
            )
        elif args.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(
                model.parameters(),
                lr=lr,
                weight_decay=args.weight_decay
            )
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=args.weight_decay
            )
        elif args.optimizer == 'radam':
            optimizer = RAdam(
                model.parameters(),
                lr=lr,
                weight_decay=args.weight_decay
            )
        return optimizer

    is_class_loss_on = False
    optimizer_func = get_optimizer(args.learning_rate)

    def calc_err(meter):
        fpr, tpr, eer = -1, -1, -1
        try:
            fpr, tpr, thresholds = roc_curve(meter.targets, meter.scores)
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            thresh = interp1d(fpr, thresholds)(eer)
        except Exception as e:
            logging.error(str(e))
            exc_type, exc_value, exc_tb = sys.exc_info()
            logging.error('\n'.join(traceback.format_exception(exc_type, exc_value, exc_tb)))
        return fpr, tpr, eer

    def forward_inner(x):
        output = None
        output_class = None
        if hasattr(model_module, 'forward_with_classification') and model.training:
            output, output_class = model.forward_with_classification(x)
        elif args.model == 'model_13_hospital':
            output, output_class = model.forward(x)

        if output is None:
            if hasattr(model_module, 'init_hidden'):
                hidden = model_module.init_hidden(batch_size=x.size(0)) #! Important // cannot be used with data parallel
                output = model.forward(x, hidden)
            else:
                output = model.forward(x)
        return output, output_class

    def process_batches(data_loader, callback):
        idx_quick_test = 0
        for batch in data_loader:
            y = batch[0].to(args.device)
            x = batch[1].to(args.device)

            output, _ = forward_inner(x)
            callback(y, x, output)

            idx_quick_test += 1
            if args.is_quick_test and idx_quick_test >= 3:
                break

    def forward(batch, centers_of_classes_by_y, is_train):
        global is_logged_cnorm, is_class_loss_on

        y = batch[0].to(args.device)
        x = batch[1].to(args.device)

        if is_train:
            if args.noise_training > 0.0:
                noise = torch.sigmoid(torch.zeros_like(x).data.normal_())
                x = args.noise_training * noise + (1.0 - args.noise_training) * x

        output, output_class = forward_inner(x)

        max_distance = 2.0 # cos
        if args.triplet_similarity != 'cos': # euclidean
            max_distance *= args.embedding_scale
            if args.triplet_similarity == 'euclidean_2': # euclidean
                max_distance = max_distance ** 2

        if args.is_triplet_loss_margin_auto:
            margin_distance = args.overlap_coef * max_distance / args.datasource_classes_train
        else:
            margin_distance = args.triplet_loss_margin

        K = args.datasource_classes_train
        C_norm = args.overlap_coef / K # max_distance included in pos_norm, neg_norm
        C_norm_center = C_norm * 0.5

        # in case of exp mining we should use normalized margin from C_norm
        if 'abs_margin' in args.filter_samples or 'abs_margin_asym' in args.filter_samples:
            margin_distance = C_norm

        sampled = triplet_sampler.sample_batch(output, y, margin_distance, max_distance)

        if not is_logged_cnorm:
            is_logged_cnorm = True
            logging.info(f'K/classes_train: {K} C_norm: {C_norm} max_distance: {max_distance}')

        pos_norm = sampled['positives_dist'] / max_distance
        neg_norm = sampled['negatives_dist'] / max_distance

        loss_pos = 0
        loss_neg = 0

        if args.triplet_loss == 'exp13':
            eps = 1e-20
            loss_pos_inner = torch.mean(torch.clamp(pos_norm - C_norm, 0.0))
            loss_neg_inner = torch.mean(torch.clamp(0.5 - neg_norm, 0.0))

            loss_pos = -torch.log(1.0 - (loss_pos_inner/(1.-C_norm)) + eps) * args.pos_loss_coef
            loss_neg = -torch.log(1.0 - (loss_neg_inner/0.5) + eps) * args.neg_loss_coef
            loss = loss_pos + loss_neg
        elif args.triplet_loss == 'standard':
            delta = sampled['positives_dist'] - sampled['negatives_dist'] + margin_distance
            if delta.size(0) == 0: # no valid triplets mined
                loss = None
            else:
                loss = torch.mean(torch.clamp(delta, min=0))
        elif args.triplet_loss == 'standard_neg_all':
            delta = sampled['positives_dist'] - sampled['negatives_dist_all_filtred'] + margin_distance
            loss = torch.mean(torch.clamp(delta, min=0))
        elif args.triplet_loss == 'lossless':
            pos = torch.mean(sampled['positives_dist'] / max_distance)
            neg = torch.mean(sampled['negatives_dist'] / max_distance)
            b = args.lossless_beta
            e = 1e-20
            loss = -torch.log10(-pos/b + 1.0 +e) - torch.log10(-(1.0-neg)/b + 1 + e)

        centers = []
        outputs_by_centers = []
        y_by_centers = []
        list_y = to_numpy(y).tolist()

        loss_emb = loss.detach().clone()
        loss_center = None
        if args.is_center_loss and args.center_loss_coef > 0:
            # centers_of_classes_by_y => all centers from current epoch
            for idx, each_y in enumerate(list_y):
                if each_y not in centers_of_classes_by_y:
                    centers_of_classes_by_y[each_y] = {
                        'sum': np.zeros((args.embedding_size, )), # numpy
                        'center': np.zeros((args.embedding_size, )), # numpy
                        'count': 0,
                        'max_dist': 0
                    }
                centers_of_classes_by_y[each_y]['sum'] += to_numpy(output[idx])
                centers_of_classes_by_y[each_y]['count'] += 1
                center = centers_of_classes_by_y[each_y]['sum'] / centers_of_classes_by_y[each_y]['count']
                centers_of_classes_by_y[each_y]['center'] = center
                if args.is_center_loss and args.center_loss_coef > 0:
                    if centers_of_classes_by_y[each_y]['count'] > args.center_loss_min_count:
                        centers.append(torch.FloatTensor(center))
                        outputs_by_centers.append(output[idx])
                        y_by_centers.append(each_y)

            if len(centers) > 0:
                centers = torch.stack(centers).to(args.device)
                outputs_by_centers = torch.stack(outputs_by_centers).to(args.device)

                centers_dist = get_distance(centers, outputs_by_centers, args.triplet_similarity, mode=args.device)
                np_centers_dist = to_numpy(centers_dist)
                for idx, each_y in enumerate(y_by_centers):
                    centers_of_classes_by_y[each_y]['max_dist'] = max(
                        centers_of_classes_by_y[each_y]['max_dist'],
                        np_centers_dist[idx]
                    )

                if args.is_center_loss:
                    if args.triplet_loss == 'exp13':
                        eps = 1e-20
                        loss_center_inner = torch.mean(torch.clamp(centers_dist - C_norm_center, 0.0))
                        loss_center = -torch.log(1.0 - (loss_center_inner/(1.-C_norm_center)) + eps)
                    else:
                        loss_center = torch.mean(torch.clamp(centers_dist - C_norm_center, 0.0))

                    loss_center = loss_center * args.center_loss_coef
                    loss += loss_center

        loss_class = None
        if is_train and args.is_class_loss and args.class_loss_coef > 0:
            if args.class_loss_epochs_limit == 0 or state['epoch'] < args.class_loss_epochs_limit:
                if output_class is not None:
                    y_hot_enc = y.to('cpu').reshape(y.size(0), 1)
                    tmp = torch.arange(args.datasource_classes_train).reshape(1, args.datasource_classes_train)
                    y_hot_enc = (y_hot_enc == tmp).float() # one hot encoded
                    y_hot_enc = y_hot_enc.detach().to(args.device)

                    loss_class = torch.mean(-y_hot_enc*torch.log(output_class))
                    loss_class = loss_class * args.class_loss_coef
                    loss += loss_class
                    is_class_loss_on = True
            else:
                if is_class_loss_on:
                    is_class_loss_on = False
                    # reset optimizer when dropping class loss
                    optimizer_func = get_optimizer(args.learning_rate)

        result = dict(
            output=output,
            output_class=output_class,
            y=y,
            loss_center=prep_loss_for_stats(loss_center),
            loss_pos=prep_loss_for_stats(loss_pos),
            loss_neg=prep_loss_for_stats(loss_neg),
            loss_class=prep_loss_for_stats(loss_class),
            loss_emb=prep_loss_for_stats(loss_emb),
            loss=loss,
            x=x
        )
        return {**result, **sampled}


    def prep_loss_for_stats(loss_val):
        if loss_val is not None:
            if isinstance(loss_val, torch.Tensor):
                loss_val = loss_val.item()
        return loss_val


    CsvUtils.create_local(args)

    meters = dict(
        train_loss = tnt.meter.AverageValueMeter(),
        test_loss = tnt.meter.AverageValueMeter(),

        train_loss_emb = tnt.meter.AverageValueMeter(),
        test_loss_emb = tnt.meter.AverageValueMeter(),

        train_loss_center = tnt.meter.AverageValueMeter(),
        test_loss_center = tnt.meter.AverageValueMeter(),

        train_loss_pos = tnt.meter.AverageValueMeter(),
        test_loss_pos = tnt.meter.AverageValueMeter(),

        train_loss_neg = tnt.meter.AverageValueMeter(),
        test_loss_neg = tnt.meter.AverageValueMeter(),

        train_loss_class = tnt.meter.AverageValueMeter(),
        test_loss_class = tnt.meter.AverageValueMeter(),

        test_acc_range = tnt.meter.ClassErrorMeter(accuracy=True),
        train_acc_range = tnt.meter.ClassErrorMeter(accuracy=True),

        test_acc_closest = tnt.meter.ClassErrorMeter(accuracy=True),
        train_acc_closest = tnt.meter.ClassErrorMeter(accuracy=True),

        test_auc_range = tnt.meter.AUCMeter(),
        train_auc_range = tnt.meter.AUCMeter(),

        test_auc_closest = tnt.meter.AUCMeter(),
        train_auc_closest = tnt.meter.AUCMeter(),

        train_dist_positives = tnt.meter.AverageValueMeter(),
        train_dist_negatives = tnt.meter.AverageValueMeter(),
        test_dist_positives = tnt.meter.AverageValueMeter(),
        test_dist_negatives = tnt.meter.AverageValueMeter(),

        train_count_positives = tnt.meter.AverageValueMeter(),
        train_count_negatives = tnt.meter.AverageValueMeter(),
        test_count_positives = tnt.meter.AverageValueMeter(),
        test_count_negatives = tnt.meter.AverageValueMeter(),

        train_count_positives_all = tnt.meter.AverageValueMeter(),
        train_count_negatives_all = tnt.meter.AverageValueMeter(),
        test_count_positives_all = tnt.meter.AverageValueMeter(),
        test_count_negatives_all = tnt.meter.AverageValueMeter(),

        train_dist_positives_hard = tnt.meter.AverageValueMeter(),
        train_dist_negatives_hard = tnt.meter.AverageValueMeter(),
        test_dist_positives_hard = tnt.meter.AverageValueMeter(),
        test_dist_negatives_hard = tnt.meter.AverageValueMeter(),
    )

    #torch.autograd.set_detect_anomaly(True)

    LoggingUtils.info(f'batch count train: {len(data_loader_train)} test: {len(data_loader_test)}')
    path_embeddings = f'{args.path_tmp_dir}/exp_loss/{args.id}'
    FileUtils.createDir(path_embeddings)

    state_before_stopping = copy.deepcopy(state)
    for epoch in range(1, args.epochs_count + 1):
        state_before = copy.deepcopy(state)
        logging.info('epoch: {} / {}'.format(epoch, args.epochs_count))

        for key in meters.keys():
            meters[key].reset()

        for data_loader in [data_loader_train, data_loader_test]:

                hist_positives_dist = []
                hist_negatives_dist = []
                hist_positives_dist_hard = []
                hist_negatives_dist_hard = []

                idx_quick_test = 0
                centers_of_classes_by_y = {
                    -1: { # example structure
                        'sum': np.zeros((args.embedding_size, )), # numpy
                        'center': np.zeros((args.embedding_size, )), # numpy
                        'count': 0,
                        'max_dist': 0
                    }
                }
                FileUtils.deleteDir(path_embeddings, is_delete_dir_path=False)

                meter_prefix = 'train'
                if data_loader == data_loader_train:
                    model = model.train()
                    torch.set_grad_enabled(True)
                else:
                    meter_prefix = 'test'
                    model = model.eval()
                    torch.set_grad_enabled(False)

                negative_max = 0

                for batch in data_loader:
                    optimizer_func.zero_grad()
                    is_train = (data_loader == data_loader_train)

                    result = forward(batch, centers_of_classes_by_y, is_train)

                    if is_train:
                        if result['loss'] is not None:
                            if result['loss'].grad_fn is not None: # in case all samples in batch does not require training
                                result['loss'].backward()
                                optimizer_func.step()

                    if result['loss'] is not None:
                        meters[f'{meter_prefix}_loss'].add(np.median(to_numpy(result['loss'])))
                    if result['loss_class'] is not None:
                        meters[f'{meter_prefix}_loss_class'].add(np.median(to_numpy(result['loss_class'])))
                    if result['loss_emb'] is not None:
                        meters[f'{meter_prefix}_loss_emb'].add(np.median(to_numpy(result['loss_emb'])))
                    if result['loss_pos'] is not None:
                        meters[f'{meter_prefix}_loss_pos'].add(np.median(to_numpy(result['loss_pos'])))
                    if result['loss_neg'] is not None:
                        meters[f'{meter_prefix}_loss_neg'].add(np.median(to_numpy(result['loss_neg'])))
                    if result['loss_center'] is not None:
                        meters[f'{meter_prefix}_loss_center'].add(np.median(to_numpy(result['loss_center'])))

                    if args.is_quick_test:
                        print(f"count pos:{float(result['positives_dist'].size(0))} neg:{float(result['negatives_dist'].size(0))}")
                        print(f"count all pos:{float(result['positives_dist_all_filtred'].size(0))} neg:{float(result['negatives_dist_all_filtred'].size(0))}")

                    meters[f'{meter_prefix}_count_positives'].add(float(result['positives_dist'].size(0)))
                    meters[f'{meter_prefix}_count_negatives'].add(float(result['negatives_dist'].size(0)))

                    meters[f'{meter_prefix}_count_positives_all'].add(float(result['positives_dist_all_filtred'].size(0)))
                    meters[f'{meter_prefix}_count_negatives_all'].add(float(result['negatives_dist_all_filtred'].size(0)))

                    if len(hist_positives_dist) < args.max_embeddings_histograms or args.max_embeddings_histograms <= 0:
                        avg_positives_dist_all = np.average(to_numpy(result['positives_dist_all']))
                        np_negatives_dist_all = to_numpy(result['negatives_dist_all'])
                        negative_max = max(negative_max, np.max(np_negatives_dist_all))
                        avg_negatives_dist_all = np.average(np_negatives_dist_all)

                        hist_positives_dist.append(avg_positives_dist_all)
                        hist_negatives_dist.append(avg_negatives_dist_all)

                        meters[f'{meter_prefix}_dist_positives'].add(avg_positives_dist_all)
                        meters[f'{meter_prefix}_dist_negatives'].add(avg_negatives_dist_all)

                        if result['positives_dist'].size(0) > 0:
                            avg_positives_dist_hard = np.average(to_numpy(result['positives_dist']))
                            meters[f'{meter_prefix}_dist_positives_hard'].add(avg_positives_dist_hard)
                            hist_positives_dist_hard.append(avg_positives_dist_hard)

                        if result['negatives_dist'].size(0) > 0:
                            avg_negatives_dist_hard = np.average(to_numpy(result['negatives_dist']))
                            hist_negatives_dist_hard.append(avg_negatives_dist_hard)
                            meters[f'{meter_prefix}_dist_negatives_hard'].add(avg_negatives_dist_hard)

                    output = to_numpy(result['output'].to('cpu')).tolist()
                    y = to_numpy(result['y']).tolist()

                    max_samples_per_class = args.max_embeddings_per_class_test
                    if data_loader == data_loader_train:
                        max_samples_per_class = args.max_embeddings_per_class_train

                    idx_quick_test += 1
                    if args.is_quick_test and idx_quick_test >= 3:
                        break

                    # END OF BACTH LOOP

                # AFTER EPOCH HISTOGRAMS
                try:

                    histogram_bins = 'auto'
                    #histogram_bins = 'doane'

                    tmp = np.array(hist_positives_dist)
                    logging.warning(f'hist_positives_dist shape: {tmp.shape} min: {np.min(tmp)} max: {np.max(tmp)}')

                    tmp = np.array(hist_negatives_dist)
                    logging.warning(f'hist_negatives_dist shape: {tmp.shape} min: {np.min(tmp)} max: {np.max(tmp)}')

                    tmp = np.array(hist_positives_dist_hard)
                    logging.warning(f'hist_positives_dist_hard shape: {tmp.shape} min: {np.min(tmp)} max: {np.max(tmp)}')

                    tmp = np.array(hist_negatives_dist_hard)
                    logging.warning(f'hist_negatives_dist_hard shape: {tmp.shape} min: {np.min(tmp)} max: {np.max(tmp)}')

                    tensorboard_writer.add_histogram(f'hist_{meter_prefix}_dist_positives', np.array(hist_positives_dist), epoch, bins=histogram_bins)
                    tensorboard_writer.add_histogram(f'hist_{meter_prefix}_dist_negatives', np.array(hist_negatives_dist), epoch, bins=histogram_bins)

                    tensorboard_writer.add_histogram(f'hist_{meter_prefix}_dist_positives_hard', np.array(hist_positives_dist_hard), epoch, bins=histogram_bins)
                    tensorboard_writer.add_histogram(f'hist_{meter_prefix}_dist_negatives_hard', np.array(hist_negatives_dist_hard), epoch, bins=histogram_bins)

                    tensorboard_utils.addHistogramsTwo(np.array(hist_positives_dist), np.array(hist_negatives_dist), f'hist_{meter_prefix}_all', epoch)
                    tensorboard_utils.addHistogramsTwo(np.array(hist_positives_dist_hard), np.array(hist_negatives_dist_hard), f'hist_{meter_prefix}_hard', epoch)

                except Exception as e:
                    logging.exception(e)

                # AFTER EPOCH CALCULATE centers of classes
                model = model.eval()
                torch.set_grad_enabled(False) # turn off gradient accumulation for speedup
                class_emb_sum = {} # for avarage center accumlation
                class_counts = {}
                class_centroids = {}
                class_max_dist = {}

                logging.info('calculate centers')
                def callback(y, x, output):
                    y = y.to('cpu').data.numpy()
                    output = output.to('cpu').data.numpy()
                    for idx_y, each_y in enumerate(y):
                        each_emb = output[idx_y]

                        if each_y not in class_emb_sum:
                            class_emb_sum[each_y] = np.zeros_like(each_emb)
                            class_counts[each_y] = 0

                        class_emb_sum[each_y] += each_emb
                        class_counts[each_y] += 1

                process_batches(data_loader, callback)

                for each_y, count_class in class_counts.items():
                    class_centroids[each_y] = class_emb_sum[each_y] / count_class
                    if args.embedding_norm == 'l2':
                        class_centroids[each_y] = normalize_vec(class_centroids[each_y])

                # AFTER EPOCH CALCULATE range of classes
                logging.info('calculate range')
                def callback(y, x, output):
                    y = y.to('cpu').data.numpy()
                    centers = [] # group together centers for quicker dist calculation
                    for idx_y, each_y in enumerate(y):
                        if each_y not in class_max_dist:
                            class_max_dist[each_y] = 0
                        centers.append(class_centroids[each_y])

                    t_centers = torch.FloatTensor(np.array(centers)).to(args.device)
                    t_dists = get_distance(output, t_centers, args.triplet_similarity, mode=args.device).to('cpu').data.numpy()
                    for idx_y, each_y in enumerate(y):
                        class_max_dist[each_y] = max(class_max_dist[each_y], t_dists[idx_y])
                process_batches(data_loader, callback)

                # AFTER EPOCH CLOSEST and range ACCURACY and store needed embeddings
                logging.info('calculate accuracy')
                list_projector_embs = []
                list_projector_imgs = []
                list_projector_labels = []
                list_projector_classes_counts = {}
                classes_size = int(np.max(list(class_centroids.keys()))) + 1

                def callback(y, x, output):
                    y = y.to('cpu').data.numpy()

                    for idx_y, each_y in enumerate(y):
                        if each_y not in class_max_dist:
                            class_max_dist[each_y] = 0

                        # first 50 classes to include in embeddings projector (y are class indexes)
                        if len(list_projector_classes_counts.keys()) < args.max_embeddings_class_for_projector:
                            if each_y not in list_projector_classes_counts:
                                list_projector_classes_counts[each_y] = 0

                        if each_y in list_projector_classes_counts:
                            # max 100 per class
                            if list_projector_classes_counts[each_y] < args.max_embeddings_projector_samples:

                                x_np = x[idx_y].to('cpu').data.numpy()
                                if x_np.min() < 0:
                                    x_np += 1.0
                                    x_np = (x_np / (x_np.max() - x_np.min()))

                                x_np *= 255
                                # C, H, W
                                x_np = x_np.swapaxes(0, 1)
                                x_np = x_np.swapaxes(1, 2)
                                # H, W, C
                                img = Image.fromarray(x_np.astype(np.uint8), mode='RGB')

                                img = img.resize((args.img_size_embeddings_class_for_projector, args.img_size_embeddings_class_for_projector), Image.ANTIALIAS)
                                img = np.array(img).astype(np.float)

                                img = img.swapaxes(2, 1)
                                img = img.swapaxes(1, 0)
                                img /= 255
                                # C, H, W

                                list_projector_imgs.append(img)
                                list_projector_embs.append(to_numpy(output[idx_y]))
                                list_projector_labels.append(data_loader.dataset.classes[each_y])

                    np_dists = np.zeros( (y.shape[0], classes_size), dtype=np.float)
                    t_centers = torch.FloatTensor(np.array(list(class_centroids.values()))).to(args.device)
                    for idx_y, each_output in enumerate(output):
                        t_each_output = each_output.repeat(t_centers.size(0), 1)
                        np_dists[idx_y] = get_distance(t_each_output, t_centers, args.triplet_similarity, mode=args.device).to('cpu').data.numpy()

                    for type in ['closest', 'range']:
                        if type == 'range':
                            predicted = np.zeros( (np_dists.shape[0], classes_size), dtype=np.float)
                        else:
                            predicted = np.ones( (np_dists.shape[0], classes_size), dtype=np.float) * 1e9

                        target = np.zeros( (np_dists.shape[0], classes_size), dtype=np.float)
                        for idx_class in range(np_dists.shape[0]):
                            target[idx_class, y[idx_class]] = 1.0

                        for each_y in class_max_dist.keys():
                            max_dist = class_max_dist[each_y]
                            if type == 'range':
                                for idx_emb, dist in enumerate(np_dists):
                                    if max_dist > dist[each_y]:
                                        predicted[idx_emb, each_y] += 1.0
                            else:
                                predicted[:,each_y] = np.minimum(predicted[:,each_y], np_dists[:, each_y]) # store for each class closest embedding with distance value

                        if type == 'range':
                            predicted = predicted / (np.sum(predicted, axis=1, keepdims=True) + 1e-18)
                        else:
                            idx_class = np.argmin(predicted, axis=1) # for each sample select closest distance
                            predicted = np.zeros_like(predicted) # init probabilities vector
                            predicted[np.arange(predicted.shape[0]), idx_class] = 1.0 # for each sample set prob 100% by columns

                        meters[f'{meter_prefix}_acc_{type}'].add(predicted, y)

                        # AssertionError: targets should be binary (0, 1)
                        idxes_classes = np.argmax(predicted, axis=1)
                        target_tp = np.array(np.equal(y, idxes_classes), dtype=np.int)
                        meters[f'{meter_prefix}_auc_{type}'].add(np.max(predicted, axis=1), target_tp)
                process_batches(data_loader, callback)

                # label_img: :math:`(N, C, H, W)
                tensorboard_writer.add_embedding(
                    mat=torch.FloatTensor(np.stack(list_projector_embs)),
                    label_img=torch.FloatTensor(np.stack(list_projector_imgs)),
                    metadata=list_projector_labels,
                    global_step=epoch, tag=f'{meter_prefix}_emb')

                state[f'{meter_prefix}_negative_max'] = negative_max
                state[f'{meter_prefix}_acc_range'] = meters[f'{meter_prefix}_acc_range'].value()[0]
                fpr, tpr, eer = calc_err(meters[f'{meter_prefix}_auc_range'])
                state[f'{meter_prefix}_eer'] = eer

                state[f'{meter_prefix}_acc_closest'] = meters[f'{meter_prefix}_acc_closest'].value()[0]
                fpr, tpr, eer = calc_err(meters[f'{meter_prefix}_auc_closest'])
                state[f'{meter_prefix}_eer2'] = eer

                if meters[f'{meter_prefix}_loss'].n > 0:
                    state[f'{meter_prefix}_loss'] = meters[f'{meter_prefix}_loss'].value()[0]

                if meters[f'{meter_prefix}_loss_emb'].n > 0:
                    state[f'{meter_prefix}_loss_emb'] = meters[f'{meter_prefix}_loss_emb'].value()[0]

                if meters[f'{meter_prefix}_loss_center'].n > 0:
                    state[f'{meter_prefix}_loss_center'] = meters[f'{meter_prefix}_loss_center'].value()[0]

                if meters[f'{meter_prefix}_loss_class'].n > 0:
                    state[f'{meter_prefix}_loss_class'] = meters[f'{meter_prefix}_loss_class'].value()[0]

                if meters[f'{meter_prefix}_loss_neg'].n > 0:
                    state[f'{meter_prefix}_loss_neg'] = meters[f'{meter_prefix}_loss_neg'].value()[0]

                if meters[f'{meter_prefix}_loss_pos'].n > 0:
                    state[f'{meter_prefix}_loss_pos'] = meters[f'{meter_prefix}_loss_pos'].value()[0]

                if meter_prefix == 'test':
                    if state[f'best_acc_closest'] < state[f'{meter_prefix}_acc_closest']:
                        state[f'best_acc_closest'] = state[f'{meter_prefix}_acc_closest']
                    if state[f'best_acc_range'] < state[f'{meter_prefix}_acc_range']:
                        state[f'best_acc_range'] = state[f'{meter_prefix}_acc_range']

                state[f'{meter_prefix}_dist_positives'] = meters[f'{meter_prefix}_dist_positives'].value()[0]
                state[f'{meter_prefix}_dist_negatives'] = meters[f'{meter_prefix}_dist_negatives'].value()[0]

                state[f'{meter_prefix}_count_positives'] = meters[f'{meter_prefix}_count_positives'].value()[0]
                state[f'{meter_prefix}_count_negatives'] = meters[f'{meter_prefix}_count_negatives'].value()[0]

                state[f'{meter_prefix}_count_positives_all'] = meters[f'{meter_prefix}_count_positives_all'].value()[0]
                state[f'{meter_prefix}_count_negatives_all'] = meters[f'{meter_prefix}_count_negatives_all'].value()[0]

                state[f'{meter_prefix}_dist_delta'] = meters[f'{meter_prefix}_dist_negatives'].value()[0] - meters[f'{meter_prefix}_dist_positives'].value()[0]

                state[f'{meter_prefix}_dist_positives_hard'] = meters[f'{meter_prefix}_dist_positives_hard'].value()[0]
                state[f'{meter_prefix}_dist_negatives_hard'] = meters[f'{meter_prefix}_dist_negatives_hard'].value()[0]

                state[f'{meter_prefix}_dist_delta_hard'] = meters[f'{meter_prefix}_dist_negatives_hard'].value()[0] - meters[f'{meter_prefix}_dist_positives_hard'].value()[0]

                tensorboard_writer.add_scalar(tag=f'{meter_prefix}_loss', scalar_value=state[f'{meter_prefix}_loss'], global_step=epoch)
                tensorboard_writer.add_scalar(tag=f'{meter_prefix}_loss_emb', scalar_value=state[f'{meter_prefix}_loss_emb'], global_step=epoch)
                tensorboard_writer.add_scalar(tag=f'{meter_prefix}_loss_center', scalar_value=state[f'{meter_prefix}_loss_center'], global_step=epoch)
                tensorboard_writer.add_scalar(tag=f'{meter_prefix}_loss_neg', scalar_value=state[f'{meter_prefix}_loss_neg'], global_step=epoch)
                tensorboard_writer.add_scalar(tag=f'{meter_prefix}_loss_pos', scalar_value=state[f'{meter_prefix}_loss_pos'], global_step=epoch)
                tensorboard_writer.add_scalar(tag=f'{meter_prefix}_loss_class', scalar_value=state[f'{meter_prefix}_loss_class'], global_step=epoch)
                tensorboard_writer.add_scalar(tag=f'{meter_prefix}_dist_delta', scalar_value=state[f'{meter_prefix}_dist_delta'], global_step=epoch)
                tensorboard_writer.add_scalar(tag=f'{meter_prefix}_dist_positives', scalar_value=state[f'{meter_prefix}_dist_positives'], global_step=epoch)
                tensorboard_writer.add_scalar(tag=f'{meter_prefix}_dist_negatives', scalar_value=state[f'{meter_prefix}_dist_negatives'], global_step=epoch)
                tensorboard_writer.add_scalar(tag=f'{meter_prefix}_dist_positives_hard', scalar_value=state[f'{meter_prefix}_dist_positives_hard'], global_step=epoch)
                tensorboard_writer.add_scalar(tag=f'{meter_prefix}_dist_negatives_hard', scalar_value=state[f'{meter_prefix}_dist_negatives_hard'], global_step=epoch)

                tensorboard_writer.add_scalar(tag=f'{meter_prefix}_count_positives', scalar_value=state[f'{meter_prefix}_count_positives'], global_step=epoch)
                tensorboard_writer.add_scalar(tag=f'{meter_prefix}_count_negatives', scalar_value=state[f'{meter_prefix}_count_negatives'], global_step=epoch)

                tensorboard_writer.add_scalar(tag=f'{meter_prefix}_count_positives_all', scalar_value=state[f'{meter_prefix}_count_positives_all'], global_step=epoch)
                tensorboard_writer.add_scalar(tag=f'{meter_prefix}_count_negatives_all', scalar_value=state[f'{meter_prefix}_count_negatives_all'], global_step=epoch)

                tensorboard_writer.add_scalar(tag=f'{meter_prefix}_negative_max', scalar_value=state[f'{meter_prefix}_negative_max'], global_step=epoch)

                tensorboard_writer.add_scalar(tag=f'{meter_prefix}_acc_range', scalar_value=state[f'{meter_prefix}_acc_range'], global_step=epoch)
                tensorboard_writer.add_scalar(tag=f'{meter_prefix}_eer', scalar_value=state[f'{meter_prefix}_eer'], global_step=epoch)
                tensorboard_writer.add_scalar(tag=f'{meter_prefix}_acc_closest', scalar_value=state[f'{meter_prefix}_acc_closest'], global_step=epoch)
                tensorboard_writer.add_scalar(tag=f'{meter_prefix}_eer2', scalar_value=state[f'{meter_prefix}_eer2'], global_step=epoch)

                tensorboard_utils.addPlot1D(
                    data=(meters[f'{meter_prefix}_auc_range'].value()[2], meters[f'{meter_prefix}_auc_range'].value()[1]),
                    tag=f'{meter_prefix}_auc_range',
                    global_step=epoch,
                    axis_labels=[
                        'False positives',
                        'True positives'
                    ]
                )
                tensorboard_utils.addPlot1D(
                    data=(meters[f'{meter_prefix}_auc_closest'].value()[2], meters[f'{meter_prefix}_auc_closest'].value()[1]),
                    tag=f'{meter_prefix}_auc_closest',
                    global_step=epoch,
                    axis_labels=[
                        'False positives',
                        'True positives'
                    ]
                )

                if data_loader == data_loader_train:
                    if args.is_reshuffle_after_epoch:
                        # after every epoch reshuffle triplets for better data mining
                        data_loader.dataset.reshuffle()

        if epoch == 1:
            state['best_param'] = state[args.early_stopping_param]
        elif state['best_param'] > state[args.early_stopping_param]:
            state['best_param'] = state[args.early_stopping_param]
            torch.save(model_module.state_dict(), os.path.join(run_path, 'best.pt'))

        epoch_time = (time.time() - time_epoch) / 60.0
        percent = epoch / args.epochs_count
        state['epoch_time'] = epoch_time

        avg_time_epochs.append(epoch_time)
        state['avg_epoch_time'] = np.average(avg_time_epochs)
        eta = ((args.epochs_count - epoch) * state['avg_epoch_time'])
        time_epoch = time.time()
        state['epoch'] = epoch

        # early stopping
        percent_improvement = 0
        if epoch > 1:
            if state_before[args.early_stopping_param] != 0:
                percent_improvement = args.early_stopping_param_coef * (state[args.early_stopping_param] - state_before[args.early_stopping_param]) / state_before[args.early_stopping_param]
                if math.isnan(percent_improvement):
                    percent_improvement = 0

            if state[args.early_stopping_param] >= 0:
                if args.early_stopping_delta_percent > percent_improvement:
                    state['early_stopping_patience'] += 1
                else:
                    state['early_stopping_patience'] = 0
            state['early_percent_improvement'] = percent_improvement

        tensorboard_writer.add_scalar(tag='improvement', scalar_value=state['early_percent_improvement'], global_step=epoch)
        torch.save(model_module.state_dict(), os.path.join(run_path, f'{args.name}.pt'))

        logging.info(
            f'{args.name} {round(percent * 100, 2)}% each: {round(state["avg_epoch_time"], 2)} min eta: {round(eta, 2)} min acc: {round(state["train_acc_range"], 2)} loss: {round(state["train_loss"], 2)} improve: {round(percent_improvement, 2)}')

        CsvUtils.add_results_local(args, state)
        CsvUtils.add_results(args, state)

        if state['early_stopping_patience'] >= args.early_stopping_patience:
            is_stop = True
            if args.learning_rate_min > 0:
                if state['learning_rate_dyn'] >= args.learning_rate_min:

                    percent_improvement = args.early_stopping_param_coef * (state[args.early_stopping_param] - state_before_stopping[args.early_stopping_param]) / state_before_stopping[args.early_stopping_param]
                    if math.isnan(percent_improvement):
                        percent_improvement = 0

                    # if improved since last stopping try
                    if percent_improvement > 0:
                        # schedule learning rate
                        state_before_stopping = copy.deepcopy(state)
                        state['learning_rate_dyn'] /= 10
                        state['early_stopping_patience'] = 0
                        optimizer_func = get_optimizer(state['learning_rate_dyn'])
                        logging_utils.info(f'{args.name} reducing LR {state["learning_rate_dyn"]}')
                        is_stop = False

            if is_stop:
                logging_utils.info(f'{args.name} early stopping')
                break

    FileUtils.deleteDir(path_embeddings, is_delete_dir_path=True)
    tensorboard_writer.close()