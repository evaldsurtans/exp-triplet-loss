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
import torchvision.transforms as transforms

class MathPlotLibUtils(object):

    @staticmethod
    def showPlot2D(dataXY):
        transform = transforms.ToPILImage()

        dataXY = np.copy(dataXY)
        min_value = np.min(dataXY)
        dataXY += abs(min_value)

        max_value = np.max(dataXY)
        dataXY /= max_value

        dataXY *= 255
        dataXY = dataXY.astype(dtype=np.uint8)

        image = np.transpose(dataXY)  # H, W
        image = np.expand_dims(image, axis=2)
        image = np.tile(image, (1, 1, 3))

        image = transform(image)
        plt.imshow(image)
        plt.show()