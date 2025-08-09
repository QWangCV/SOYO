import os
import sys
import copy
import time
import torch
import logging
import numpy as np
from utils import factory
from utils.data_manager import DataManager


def train(args):
    seed_list = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])

    for seed in seed_list:
        args['seed'] = seed
        args['device'] = device
        _train(args)


def get_remaining_space(folder_path):
    try:
        disk_info = os.statvfs(folder_path)
        block_size = disk_info.f_frsize
        available_blocks = disk_info.f_bavail
        remaining_space = available_blocks * block_size
        return remaining_space
    except Exception as e:
        print("Error:", e)
        return None


def _train(args):
    ### log
    # set log path
    log_path = os.path.join(args['log_path'], args['dataset'])
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # check the remaining space
    remaining_space = get_remaining_space(log_path)
    if remaining_space is not None:
        remaining_space_gb = remaining_space / (1024 ** 3)  # Convert bytes to gigabytes
        print(f"Remaining space in {log_path}: {remaining_space_gb:.2f} GB")
        if remaining_space_gb < 0.5:
            raise OSError('Remaining space is not enough to store checkpoint.')
    
    # set run name and log
    run_name = f"{args['net_type']}_{args['dataset']}"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] %(message)s',
        handlers=[logging.FileHandler(filename = os.path.join(log_path, run_name+'.log')),
                  logging.StreamHandler(sys.stdout)],
        datefmt = '%Y-%m-%d %H:%M:%S'
    )
    logging.info(f'Run Name: {run_name}')
    _set_random()
    _set_device(args)
    print_args(args)
    
    ### dataset
    data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'], args)
    args['class_order'] = data_manager._class_order
    
    ### model
    model = factory.get_model(args['model_name'], args)
    cnn_curve = {'top1': []}

    for task in range(data_manager.nb_tasks):
        model.incremental_train(data_manager)
        cnn_accy = model.eval_task()
        
        logging.info('CNN: {}'.format(cnn_accy['grouped']))
        cnn_curve['top1'].append(cnn_accy['top1'])
        logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
        
        model.after_task()  # domain += 1

        if not os.path.exists(os.path.join(log_path, run_name)):
            os.makedirs(os.path.join(log_path, run_name))
        torch.save(model, os.path.join(log_path, run_name, f"task_{int(task)}.pth"))
        logging.info(f'Save the checkpoint task_{int(task)}.pth')
    


def _set_device(args):
    device_type = args['device']
    gpus = []
    for device in device_type:
        if device == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))
        gpus.append(device)
    args['device'] = gpus


def _set_random():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))
