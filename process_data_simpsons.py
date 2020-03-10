import argparse
import math
import random
import time
import json
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures

from PIL import Image
# pip install resize-and-crop
from resize_and_crop import resize_and_crop

from modules.file_utils import FileUtils
from modules.logging_utils import LoggingUtils

parser = argparse.ArgumentParser(description='Process Kaggle The Simpsons to memmap for dataset')

# /the-simpsons-characters-dataset/kaggle_simpson_testset/kaggle_simpson_testset/abraham_grampa_simpson_1.jpg
parser.add_argument('-path_input_test', default='/Users/evalds/Desktop/the-simpsons-characters-dataset/kaggle_simpson_testset/kaggle_simpson_testset', type=str)

# /the-simpsons-characters-dataset/simpsons_dataset/abraham_grampa_simpson/pic_0001.jpg
parser.add_argument('-path_input_train', default='/Users/evalds/Desktop/the-simpsons-characters-dataset/simpsons_dataset', type=str)

# /simpsons/test.mmap
# /simpsons/test.json
parser.add_argument('-path_output', default='/Users/evalds/Downloads/simpsons_x/', type=str)

# scale and squeeze images to this size
parser.add_argument('-size_img', default=128, type=int)
parser.add_argument('-thread_max', default=10, type=int)

parser.add_argument('-test_split', default=0.2, type=float)

args, args_other = parser.parse_known_args()

FileUtils.createDir(args.path_output)
logging_utils = LoggingUtils(f"{args.path_output}/simpsons-{datetime.now().strftime('%y-%m-%d_%H-%M-%S')}.log")

class_names = []
last_class_name = None

mmap_shape = [0, 3, args.size_img, args.size_img]

logging_utils.info(f'move test samples into train to change from classification to re-identification task')

paths_files = FileUtils.listSubFiles(args.path_input_test)
for path_file in paths_files:
    base_path_file = os.path.basename(path_file)
    base_path_file = base_path_file[:-4] # remove .jpg at the end
    str_sample_idx = base_path_file[base_path_file.rindex('_')+1:]
    base_path_file = base_path_file[:base_path_file.rindex('_')] # remove sample idx like _22
    if os.path.exists(f'{args.path_input_train}/{base_path_file}'):
        os.rename(path_file, f'{args.path_input_train}/{base_path_file}/test_{str_sample_idx}.jpg')
    else:
        LoggingUtils.error(f'not exiting, cannot move from test: {args.path_input_train}/{base_path_file}')
        exit()

logging_utils.info(f'samples started to gather')

samples_groups_by_class = {}
samples_count = 0

dir_person_ids = os.listdir(args.path_input_train)
for person_id in dir_person_ids:
    path_person_id = f'{args.path_input_train}/{person_id}'
    if os.path.isdir(path_person_id):

        dir_images = os.listdir(path_person_id)
        for path_image_each in dir_images:
            path_image = f'{path_person_id}/{path_image_each}'

            if os.path.isfile(path_image):
                if last_class_name != person_id:
                    last_class_name = person_id
                    class_names.append(person_id)

                class_idx = len(class_names) - 1
                if class_idx not in samples_groups_by_class:
                    samples_groups_by_class[class_idx] = []

                samples_groups_by_class[class_idx].append(path_image)
                samples_count += 1


logging_utils.info(f'samples gathered: {samples_count}')


idxes_shuffled = np.arange(len(class_names), dtype=np.int).tolist()
random.shuffle(idxes_shuffled)
class_names_all = np.array(class_names)[idxes_shuffled]

class_names_test = class_names_all[:int(len(class_names_all) * args.test_split)]
class_names_train = class_names_all[len(class_names_test):]


for base_name, class_names_include in [
    ('train', class_names_train),
    ('test', class_names_test)
]:
    class_names_include = class_names_include.tolist()
    samples_by_class_idxes = []
    samples_by_paths = []

    for class_name in class_names_include:
        class_idx_local = class_names_include.index(class_name)
        class_idx_global = class_names.index(class_name)
        samples = samples_groups_by_class[class_idx_global]
        for sample_path in samples:
            samples_by_paths.append((len(samples_by_paths), sample_path))
            samples_by_class_idxes.append(class_idx_local)

    mmap_shape[0] = len(samples_by_paths)
    mem = np.memmap(
        f'{args.path_output}/{base_name}.mmap',
        mode='w+',
        dtype=np.float16,
        shape=tuple(mmap_shape))
    print(f'mmap_shape: {mmap_shape}')

    with open(f'{args.path_output}/{base_name}.json', 'w') as fp:
        json.dump({
            'class_names': class_names_include,
            'mmap_shape': mmap_shape,
            'samples_by_class_idxes': samples_by_class_idxes
        }, fp, indent=4)

    logging_utils.info('finished json')


    def thread_processing(sample):
        idx_sample, path_image = sample
        idx_sample = int(idx_sample)
        if idx_sample % 1000 == 0:
            logging_utils.info(f'idx_sample: {idx_sample}/{mmap_shape[0]}')
        img = Image.open(path_image)
        if img.mode != "RGB":
            img = img.convert("RGB")

        if img.size[0] > img.size[1]:
            # wider than taller
            ratio = img.size[1] / img.size[0]
            img = img.resize((args.size_img, int(round(args.size_img * ratio))), Image.ANTIALIAS)
        elif img.size[0] < img.size[1]:
            # portrait
            ratio = img.size[0] / img.size[1]
            img = img.resize((int(round(args.size_img * ratio)), args.size_img), Image.ANTIALIAS)
        else:
            img = img.resize((args.size_img, args.size_img), Image.ANTIALIAS)

        np_image = np.zeros((args.size_img, args.size_img, 3))
        np_image_part = np.array(img)

        pad_left = int((args.size_img - img.size[0]) * 0.5)
        pad_right = int(math.ceil((args.size_img - img.size[0]) * 0.5))
        pad_top = int((args.size_img - img.size[1]) * 0.5)
        pad_bottom = int(math.ceil((args.size_img - img.size[1]) * 0.5))

        np_image[pad_top:args.size_img-pad_bottom, pad_left:args.size_img-pad_right] = np_image_part

        np_image = np.swapaxes(np_image, 1, 2)
        np_image = np.swapaxes(np_image, 0, 1)
        np_image /= 255
        #np_image -= 0.5
        #np_image *= 2

        # for debug
        # np_image = 0.2989 * np_image[0, :] + 0.5870 * np_image[1, :] + 0.1140 * np_image[2, :]
        # plt.imshow(np_image)
        # plt.show()

        mem[idx_sample] = np_image

    time_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.thread_max) as executor:
        executor.map(thread_processing, samples_by_paths)
    logging_utils.info(f'done in: {(time.time() - time_start)/60} min')

    mem.flush()

    logging_utils.info('finished processing')



