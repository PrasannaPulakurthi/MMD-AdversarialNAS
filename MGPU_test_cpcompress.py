from __future__ import absolute_import, division, print_function

import cfg_compress
import archs
import datasets
from network import train, validate, LinearLrDecay, load_params, copy_params
from utils.utils import set_log_dir, save_checkpoint, create_logger, count_parameters_in_MB
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception
from utils.flop_benchmark import print_FLOPs
from utils.compress_utils import *

import torch
import os
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from copy import deepcopy
from decompose import Compression, DecompositionInfo, CompressionInfo
from utils.metrics import PerformanceStore

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main():
    args = cfg_compress.parse_args()
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # set visible GPU ids
    if len(args.gpu_ids) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        
    # set TensorFlow environment for evaluation (calculate IS and FID)
    _init_inception([args.eval_batch_size,args.img_size,args.img_size,3])
    inception_path = check_or_download_inception('./tmp/imagenet/')
    create_inception_graph(inception_path)

    # the first GPU in visible GPUs is dedicated for evaluation (running Inception model)
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for id in range(len(str_ids)):
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 1:
      args.gpu_ids = args.gpu_ids[1:]
    else:
      args.gpu_ids = args.gpu_ids
    
    # genotype G
    genotypes_root = os.path.join('exps', args.genotypes_exp, 'Genotypes')
    genotype_G = np.load(os.path.join(genotypes_root, 'latest_G.npy'))

    # genotype D
    # genotype_D = np.load(os.path.join(genotypes_root, 'latest_D.npy'))

    # import network from genotype
    basemodel_gen = eval('archs.' + args.arch + '.Generator')(args, genotype_G)
    gen_net = torch.nn.DataParallel(basemodel_gen, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])
    # basemodel_dis = eval('archs.' + args.arch + '.Discriminator')(args)
    # dis_net = torch.nn.DataParallel(basemodel_dis, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])

    # fid stat
    if args.dataset.lower() == 'cifar10':
        fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
    elif args.dataset.lower() == 'stl10':
        fid_stat = 'fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
    elif args.dataset.lower() == 'celeba':
        fid_stat = 'fid_stat/fid_stats_celeba_hq_256.npz'
    else:
        raise NotImplementedError(f'no fid stat for {args.dataset.lower()}')
    assert os.path.exists(fid_stat)
    
    # initial
    gen_avg_param = copy_params(gen_net)
    start_epoch = 0
    best_fid = 1e4
    best_is = 0

    # model size
    print('Model Size for the Uncompressed Network')
    gen_params0 = count_parameters_in_MB(gen_net)
    # dis_params0 = count_parameters_in_MB(dis_net)
    gen_flops0 = print_FLOPs(basemodel_gen, (1, args.latent_dim))
    # dis_flops0 = print_FLOPs(basemodel_dis, (1, 3, args.img_size, args.img_size))

    print('Initial Param size of G = ', gen_params0, 'M')
    # print('Initial Param size of D = ', count_parameters_in_MB(dis_net), 'M')

    # Instanciate the Compression object
    compress_obj = Compression(gen_params0, gen_flops0)


    # resuming
    print(f'=> resuming from {args.checkpoint}')
    print(os.path.join('exps', args.checkpoint))
    assert os.path.exists(os.path.join('exps', args.checkpoint))
    checkpoint_file = os.path.join('exps', args.checkpoint, 'Model', 'checkpoint_best.pth')
    assert os.path.exists(checkpoint_file)
    checkpoint = torch.load(checkpoint_file)

    start_epoch = checkpoint['epoch']
    best_fid = checkpoint['best_fid']
    try:
        best_is = checkpoint['best_is']
    except:
        best_is = 0

    if 'decomposition_info' in checkpoint.keys():
        print('Applying decomposition to generator architecture from the checkpoint...')
        try:
            compression_info = checkpoint['compression_info']
        except:
            compression_info = None
        compress_obj.apply_decomposition_from_checkpoint(args, gen_net, checkpoint['decomposition_info'], compression_info, replace_only=True)  # apply decomposition before loading checkpoint
    else:
        # starting from pretrained model
        # re-set the best_fid and best_is, otherwise, 
        # the best checkpoint will not be saved due to 
        # the performance degradation caused by the compression
        args.resume = False ## saves to different folders
        best_fid = 1e4
        best_is = 0
        start_epoch = 0
    
    if 'performance_store' in checkpoint.keys():
        performance_store = checkpoint['performance_store']
        print('Loaded performance store from the checkpoint')
        print(performance_store)
    else:
        performance_store = None
    
    gen_net.load_state_dict(checkpoint['gen_state_dict'])
    # dis_net.load_state_dict(checkpoint['dis_state_dict'])
    avg_gen_net = deepcopy(gen_net)
    avg_gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
    gen_avg_param = copy_params(avg_gen_net)
    del avg_gen_net

    if args.resume:
        args.path_helper = checkpoint['path_helper']
    else:
        args.path_helper = set_log_dir('exps', args.exp_name)

    logger = create_logger(args.path_helper['log_path'])
    logger.info(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'valid_global_steps': start_epoch // args.val_freq,
    }
    
    # model size
    logger.info('Param size of G = %fMB', count_parameters_in_MB(gen_net))
    # logger.info('Param size of D = %fMB', count_parameters_in_MB(dis_net))
    # print_FLOPs(basemodel_gen, (1, args.latent_dim), logger)
    # print_FLOPs(basemodel_dis, (1, 3, args.img_size, args.img_size), logger)

    if performance_store is None:
        performance_store = PerformanceStore()
    
    # for visualization
    if args.draw_arch:
        from utils.genotype import draw_graph_G, draw_graph_D
        draw_graph_G(genotype_G, save=True, file_path=os.path.join(args.path_helper['graph_vis_path'], 'latest_G'))
        # draw_graph_D(genotype_D, save=True, file_path=os.path.join(args.path_helper['graph_vis_path'], 'latest_D'))

    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (100, args.latent_dim)))

    load_params(gen_net, gen_avg_param)
    inception_score, std, fid_score = validate(args, fixed_z, fid_stat, gen_net, writer_dict)
    logger.info(f'Initial Inception score mean: {inception_score}, Inception score std: {std}, '
                f'FID score: {fid_score} || @ epoch {start_epoch}.')
    
if __name__ == '__main__':
    main()
