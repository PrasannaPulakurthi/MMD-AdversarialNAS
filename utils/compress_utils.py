
import os
import torch
import dateutil.tz
from datetime import datetime
import time
import logging

import numpy as np

import matplotlib.pyplot as plt


def validate_args(args):
    layers = args.layers
    if layers is None:
        print('No layers specified.')
        args.layers = []
        args.ranks = []
        return
    
    assert(isinstance(layers,list))
    N = len(layers)

    if args.byrank:
        if args.rank is None:
            raise ValueError('rank must be specified when by_rank is True')
        elif len(args.rank) == 1:
            args.rank = args.rank * N
        elif len(args.rank) != N:
            raise ValueError('rank must be a list of length N')
    elif args.byratio:
        if args.compress_ratio is None or args.compress_ratio<0 or args.compress_ratio>1:
            raise ValueError('compress_ratio must be specified and in range [0,1]')
def validate_dis_rgs(args):
    layers = args.dis_layers
    if layers is None:
        print('No layers specified.')
        args.dis_layers = []
        args.dis_rank = []
        return
    
    assert(isinstance(layers,list))
    N = len(layers)

    if args.byrank:
        if args.dis_rank is None:
            raise ValueError('rank must be specified when by_rank is True')
        elif len(args.dis_rank) == 1:
            args.dis_rank = args.dis_rank * N
        elif len(args.dis_rank) != N:
            raise ValueError('rank must be a list of length N')
    elif args.byratio:
        if args.compress_ratio is None or args.compress_ratio<0 or args.compress_ratio>1:
            raise ValueError('compress_ratio must be specified and in range [0,1]')
    
def set_root_dir(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)
    # set log path
    exp_path = os.path.join(root_dir, exp_name)

    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix)

    path_dict['prefix'] = prefix

    log_path=os.path.join(prefix,'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    return path_dict
    



def set_step_dir(path_helper, step, layer, rank):
    root_dir = path_helper['prefix']

    path_dict = {}
    
    prefix = os.path.join(root_dir, 'step_{}_{}_{}'.format(step,layer,rank))
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    return path_dict


def calculate_rank_for_ratio(layer, ratio):
    shape = torch.tensor(layer.weight.shape)
    R = (ratio*torch.prod(shape).item()) / torch.sum(shape).item()
    return max(1, int(R))

def get_rank_for_layer_by_name(model, layer_name, ratio):
    for name, l in model.named_modules():
        if name == layer_name:
            return calculate_rank_for_ratio(l, ratio)
    raise ValueError(f'layer {layer_name} not found in model')

def get_ranks_per_layer(model, ratio, layers):
    ranks = []
    for name in layers:
        R = get_rank_for_layer_by_name(model, name, ratio)
        ranks.append(R)
    return ranks


def plot_performance(performance_dict, dir_):
    if 'e' in performance_dict.keys():
        x_ax = performance_dict['e']
    else:
        x_ax = np.range(len(performance_dict['fid']))
        
    for name, performance in performance_dict.items():
        if name == 'e':
            continue
        f=plt.figure()
        plt.plot(performance, '-o', label=name)
        plt.plot([0, len(performance)-1],[performance[0], performance[0]], '--k', label='initial', alpha=0.5 )
        plt.legend()
        plt.xlabel('step')
        plt.ylabel(name)
        plt.savefig(os.path.join(dir_, name+'.png'))
        plt.close(f)
