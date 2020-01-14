import multiprocessing
import shlex

from sklearn.model_selection import ParameterGrid
import subprocess
import traceback
import logging
import os
import re
import sys
import time
import copy
import json
import platform
import numpy as np
import argparse
from datetime import datetime
from modules.logging_utils import LoggingUtils
from modules.args_utils import ArgsUtils
from sklearn.model_selection import ParameterGrid
import subprocess

from modules.file_utils import FileUtils
import typing

parser = argparse.ArgumentParser(description='task generator linux local for optimal use of resources', add_help=False)

parser.add_argument(
    '-repeat',
    help='how many times each set of parameters should be repeated for testing stability',
    default=1,
    type=int)

parser.add_argument(
    '-report',
    help='csv report of all tasks combined',
    default='tasks',
    type=str)

parser.add_argument(
    '-single_task',
    help='for testing generate only single task (debug)',
    default=False,
    type=lambda x: (str(x).lower() == 'true'))

parser.add_argument(
    '-conda_env',
    help='name of conda environment',
    default='conda_env',
    type=str)

parser.add_argument(
    '-params_grid',
    nargs='*',
    help='parameters for grid search',
    required=False) # for search

parser.add_argument(
    '-local_process_count_per_task',
    help='how many simulations/parallel tasks should be run per local PC task',
    default=1,
    type=int)

parser.add_argument(
    '-tf_path_train',
    nargs='*',
    required=False)

parser.add_argument(
    '-tf_ratio_train',
    nargs='*',
    required=False)

parser.add_argument(
    '-tf_path_test',
    nargs='*',
    required=False)

parser.add_argument(
    '-params_report',
    help='csv columns, parameters for summary',
    default=['id', 'name', 'repeat_id'],
    nargs='*',
    required=False) # extra params for report header

parser.add_argument(
    '-is_single_cuda_device',
    help='split tasks for single gpu',
    default=True,
    type=lambda x: (str(x).lower() == 'true'))

parser.add_argument(
    '-cuda_devices_used',
    default=[],
    nargs='*',
    required=False)

parser.add_argument(
    '-is_force_start',
    default=False,
    type=lambda x: (str(x).lower() == 'true'))

parser.add_argument(
    '-is_restricted_memory',
    default=False,
    type=lambda x: (str(x).lower() == 'true'))

parser.add_argument(
    '-is_restricted_cpu',
    default=False,
    type=lambda x: (str(x).lower() == 'true'))


args, args_other = parser.parse_known_args()
args = ArgsUtils.add_other_args(args, args_other)
args_other_names = ArgsUtils.extract_other_args_names(args_other)

if args.is_restricted_cpu:
    from cgroups import Cgroup # pip install cgroups
    # sudo /home/ubuntu/anaconda3/bin/user_cgroups ubuntu
    # sudo /home/evalds/.conda/envs/conda_env/bin/user_cgroups evalds

# add all testable parameters to final report header
args.params_report += args_other_names

FileUtils.createDir('./reports')
FileUtils.createDir('./tasks')
FileUtils.createDir('./tasks/' + args.report)

logging_utils = LoggingUtils(filename=os.path.join('reports', args.report + '.txt'))
ArgsUtils.log_args(args, 'taskgen.py', logging_utils)

task_settings = {
    'id': 0,
    'repeat_id': 0
}
tasks_settings_path = os.path.join('tasks', 'tasks.json')
if os.path.exists(tasks_settings_path):
    with open(tasks_settings_path, 'r') as outfile:
        tasks_settings_loaded = json.load(outfile)
        for key in tasks_settings_loaded:
            task_settings[key] = tasks_settings_loaded[key]

formated_params_grid = {}
formated_params_seq = {}

if not args.params_grid is None:
    for key_grid in args.params_grid:
        formated_params_grid[key_grid] = []

        for arg in vars(args):
            key = arg
            value = getattr(args, arg)

            if key == key_grid:
                if value is None:
                    raise Exception('Missing values for grid search key: {}'.format(key_grid))
                if len(value) < 2:
                    raise Exception('Not enough grid search values for key: {}'.format(key_grid))
                else:
                    formated_params_grid[key_grid] += value
                break


for arg in vars(args):
    key = arg
    value = getattr(args, arg)

    if key in args_other_names:
        if not key in formated_params_grid:
            if not value is None and len(value) > 0 and not value[0] is None:
                formated_params_seq[key] = value

logging.info(formated_params_seq)


grid = []
if len(list(formated_params_grid)) > 0:
    grid = list(ParameterGrid(formated_params_grid))

# add sequences
for each_seq in formated_params_seq:
    if len(formated_params_seq[each_seq]) > 1:
        for value in formated_params_seq[each_seq]:
            grid.append({
                each_seq: value
            })

if len(grid) == 0:
    grid.append({})

# add const params
for each_seq in formated_params_seq:
    value = formated_params_seq[each_seq]
    if len(value) == 1:
        for each_grid in grid:
            each_grid[each_seq] = value[0]

path_base = os.path.dirname(os.path.abspath(__file__))

tmp = ['id', 'name', 'repeat_id']
if not args.params_report is None:
    for it in args.params_report:
        if not it in tmp:
            tmp.append(it)
args.params_report = tmp

any_count = 0
max_count = len(grid) * args.repeat
logging_utils.info('{} total tasks {}'.format(args.report, max_count))

script_path = ''
process_per_task = 0


logging_utils.info('tasks summary:')
tmp_id = task_settings['repeat_id']
tmp_cnt = 0
if args.single_task:
    grid = grid[:1]

for params_comb in grid:
    tmp_id += 1
    tmp_cnt += 1
    params_comb['tf_path_test'] = args.tf_path_test
    params_comb['tf_path_train'] = args.tf_path_train
    params_comb['tf_ratio_train'] = args.tf_ratio_train

    tmp_serialized = json.dumps(params_comb, indent=4)
    logging_utils.info(f'\n\n{tmp_cnt} / {len(grid)}: {tmp_id}')
    logging_utils.info(tmp_serialized)

logging.info(f'formated_params_grid:{json.dumps(formated_params_grid, indent=4)}')

if not args.is_force_start:
    print('are tests ok? proceed?')
    if input('[y/n]: ') != 'y':
        exit()

windows_log_list = []

cuda_device_id = 0
cuda_devices_max = 0

is_windows = False
script_ext = '.sh'
if platform.system().lower() == 'windows':
    script_ext = '.bat'
    is_windows = True

try:
    cuda_devices_max = len(list(os.listdir('/proc/driver/nvidia/gpus/')))

    # use all in this case
    if len(args.cuda_devices_used) == 0:
        args.cuda_devices_used = np.arange(0, cuda_devices_max).tolist()
except:
    logging.error('cuda_devices_max not found')

cuda_devices_max = len(args.cuda_devices_used)

multiprocessing.set_start_method('spawn', force=True)
parallel_processes = []

cuda_devices_in_use = [{ 'idx': idx, 'tasks': []} for idx in range(cuda_devices_max)]

if args.is_restricted_cpu:
    # sudo /home/asya/anaconda3/envs/conda_env/bin/user_cgroups asya

    # Limit resources
    # SC_PHYS_PAGES, SC_AVPHYS_PAGES
    cg = Cgroup('task_gener')
    cpu_process = int(90 / args.local_process_count_per_task)
    cg.set_cpu_limit(cpu_process) # %
    cg.set_memory_limit(limit=None)

max_mem_process = 0
if args.is_restricted_memory:
    max_ram_bytes_available = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_AVPHYS_PAGES')
    max_mem_process = int(max_ram_bytes_available * 0.9 / args.local_process_count_per_task)  # with 10% in reserve

    # cannot use cgroup to limit RAM, because when it is over the limit it will kill process not limit it`s RAM usage
    #cg.set_memory_limit(max_ram_bytes_available, unit='bytes')

def on_preexec_fn():
    if args.is_restricted_cpu:
        pid = os.getpid()
        cg.add(pid)

for idx_comb, params_comb in enumerate(grid):
    task_settings['repeat_id'] += 1

    params_comb['tf_data_workers'] = 1 # very important to manage mmap via ulimit
    params_comb['report'] = args.report
    params_comb['id'] = args.report
    params_comb['params_report'] = ' '.join(args.params_report)
    params_comb['repeat_id'] = task_settings['repeat_id']

    for idx_repeat in range(args.repeat):
        task_settings['id'] += 1

        with open(tasks_settings_path, 'w') as outfile:
            json.dump(task_settings, outfile)

        params_comb['id'] = task_settings['id']
        params_comb['name'] = args.report + '_' + str(task_settings['repeat_id']) + '_' + str(task_settings['id'])

        str_params = []
        for key in params_comb:
            value_param = params_comb[key]
            if isinstance(value_param, typing.List):
                value_param = ' '.join(value_param)
            str_params.append('-' + key + ' ' + str(value_param))

        if max_mem_process > 0:
           str_params.append(f'-max_mem_process {max_mem_process}')
        str_params = ' '.join(str_params)

        script_path = f'{path_base}/tasks/{args.report}/' + params_comb['name'] + script_ext
        with open(script_path, 'w') as fp:
            fp.write(f'#!/bin/bash -v\n')
            if args.conda_env != '' and args.conda_env != 'none':
                fp.write(f'source activate {args.conda_env}\n')

            fp.write(f'ulimit -n 500000\n')

            if args.is_restricted_memory:
                fp.write(f'ulimit -Sm {int(max_mem_process*0.5/1000)}\n')
                fp.write(f'ulimit -Hm {int(max_mem_process/1000)}\n')

            prefix_cuda = ''
            if args.is_single_cuda_device:

                # find most free cuda device
                cuda_devices_in_use = sorted(cuda_devices_in_use, key=lambda it: len(it['tasks']), reverse=False)

                cuda_device_id = args.cuda_devices_used[cuda_devices_in_use[0]['idx']]
                prefix_cuda = f'CUDA_VISIBLE_DEVICES={cuda_device_id} '

                cuda_devices_in_use[0]['tasks'].append(script_path)
            else:
                if len(args.cuda_devices_used) > 0:
                    prefix_cuda = f'CUDA_VISIBLE_DEVICES={",".join(args.cuda_devices_used)} '

            fp.write(f'{prefix_cuda}python {path_base}/main.py {str_params}\n')

        cmd = f'chmod +x {script_path}'
        stdout = subprocess.call(shlex.split(cmd))

        any_count += 1
        logging.info('\n\n\n\nTask: {} / {}'.format(any_count, max_count))
        logging.info(script_path)

        process = subprocess.Popen(script_path, shell=False, preexec_fn=on_preexec_fn)
        process.script_path = script_path
        parallel_processes.append(process)

        while len(parallel_processes) >= args.local_process_count_per_task:
            time.sleep(1)
            parallel_processes_filtred = []
            for process in parallel_processes:
                if process.poll() is not None:
                    if args.is_single_cuda_device:
                        for it in cuda_devices_in_use:
                            if process.script_path in it['tasks']:
                                it['tasks'].remove(process.script_path)
                else:
                    parallel_processes_filtred.append(process)
            parallel_processes = parallel_processes_filtred

        if args.single_task:
            logging.info('Single task test mode completed')
            exit()