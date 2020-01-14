import argparse
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

from modules.logging_utils import LoggingUtils

parser = argparse.ArgumentParser(description='Process VGG to memmap for dataset')

# /vggface2/test/n009291/0002_01.jpg
parser.add_argument('-path_input', default='/Users/evalds/Downloads/vggface2/test', type=str)

# /vggface2/test.mmap
# /vggface2/test.json
parser.add_argument('-path_output', default='/Users/evalds/Downloads/vggface2/', type=str)

parser.add_argument('-size_img', default=128, type=int)
parser.add_argument('-thread_max', default=200, type=int)
parser.add_argument('-is_only_json', default=False, type=lambda x: (str(x).lower() == 'true'))

args, args_other = parser.parse_known_args()

base_name = os.path.basename(args.path_input)
logging_utils = LoggingUtils(f"{args.path_output}/{base_name}-{datetime.now().strftime('%y-%m-%d_%H-%M-%S')}.log")

class_names = []
samples_by_class_idxes = []
samples_by_paths = []
last_class_name = None

mmap_shape = [0, 3, args.size_img, args.size_img]

logging_utils.info(f'samples started to gather')

dir_person_ids = os.listdir(args.path_input)
for person_id in dir_person_ids:
    path_person_id = f'{args.path_input}/{person_id}'
    if os.path.isdir(path_person_id):

        dir_images = os.listdir(path_person_id)
        for path_image_each in dir_images:
            path_image = f'{path_person_id}/{path_image_each}'

            if os.path.isfile(path_image):
                if last_class_name != person_id:
                    last_class_name = person_id
                    class_names.append(person_id)

                class_idx = len(class_names) - 1
                samples_by_class_idxes.append(class_idx)
                samples_by_paths.append((len(samples_by_paths), path_image))

logging_utils.info(f'samples gathered: {len(samples_by_class_idxes)}')


mmap_shape[0] = len(samples_by_paths)
mem = np.memmap(
    f'{args.path_output}/{base_name}.mmap',
    mode='w+',
    dtype=np.float16,
    shape=tuple(mmap_shape))


def thread_processing(sample):
    idx_sample, path_image = sample
    if idx_sample % 1000 == 0:
        logging_utils.info(f'idx_sample: {idx_sample}/{mmap_shape[0]}')
    #image = Image.open(path_image)
    image = resize_and_crop(path_image, (args.size_img, args.size_img), "middle")
    np_image = np.array(image.getdata()).reshape(args.size_img, args.size_img, 3)
    np_image = np.swapaxes(np_image, 1, 2)
    np_image = np.swapaxes(np_image, 0, 1)
    mem[idx_sample] = np_image

time_start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=args.thread_max) as executor:
    executor.map(thread_processing, samples_by_paths)
logging_utils.info(f'done in: {(time.time() - time_start)/60} min')

mem.flush()

logging_utils.info('finished processing')

with open(f'{args.path_output}/{base_name}.json', 'w') as fp:
    json.dump({
        'class_names': class_names,
        'mmap_shape': mmap_shape,
        'samples_by_class_idxes': samples_by_class_idxes
    }, fp, indent=4)

logging_utils.info('finished json')

