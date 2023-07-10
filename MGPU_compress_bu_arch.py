# @Date    : 2019-10-22
# @Author  : Chen Gao

from __future__ import absolute_import, division, print_function

import cfg_compress
import archs
import datasets
from network import train, validate, LinearLrDecay, load_params, copy_params
from utils.utils import set_log_dir, save_checkpoint, create_logger, count_parameters_in_MB
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception
from utils.flop_benchmark import print_FLOPs

import torch
import os
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from copy import deepcopy

import pathlib

from utils.compress_utils import * #validate_args, set_root_dir, set_step_dir, get_rank_for_layer_by_names

from decompose import *
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def main():
    args = cfg_compress.parse_args()
    validate_args(args)
    torch.cuda.manual_seed(args.random_seed)

    str_ = ''
    if args.byrank:
        str_+= 'byrank'+str(args.rank[0])
    else:
        str_+= 'byratio'+str(args.compress_ratio)
    args.exp_name = 'CP-compress-'+args.compress_mode+'-'+args.dataset + '-'+ str_+'-' + args.exp_name

    
    # set visible GPU ids
    if len(args.gpu_ids) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        
    # set TensorFlow environment for evaluation (calculate IS and FID)
    _init_inception()
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

    # import network from genotype
    basemodel_gen = eval('archs.' + args.arch + '.Generator')(args, genotype_G)
    gen_net = torch.nn.DataParallel(basemodel_gen, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])
    basemodel_dis = eval('archs.' + args.arch + '.Discriminator')(args)
    dis_net = torch.nn.DataParallel(basemodel_dis, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])

    # weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
            
    gen_net.apply(weights_init)
    dis_net.apply(weights_init)
    
    # set up data_loader
    dataset = datasets.ImageDataset(args)
    train_loader = dataset.train
    
    # epoch number for dis_net
    args.max_epoch_D = args.max_epoch_G * args.n_critic
    if args.max_iter_G:
        args.max_epoch_D = np.ceil(args.max_iter_G * args.n_critic / len(train_loader))
    max_iter_D = args.max_epoch_D * len(train_loader)
    
    # set optimizer
    gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                     args.g_lr, (args.beta1, args.beta2))
    dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                     args.d_lr, (args.beta1, args.beta2))
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, max_iter_D)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, max_iter_D)

    # fid stat
    if args.dataset.lower() == 'cifar10':
        fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
    elif args.dataset.lower() == 'stl10':
        fid_stat = 'fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
    else:
        raise NotImplementedError(f'no fid stat for {args.dataset.lower()}')
    assert os.path.exists(fid_stat)
    
    # initial
    gen_avg_param = copy_params(gen_net)
    start_epoch = 0
    best_fid = 1e4
    best_is = 0 

    # set writer
    if args.checkpoint:
        # resuming
        print(f'=> resuming from {args.checkpoint}')
        print(os.path.join('exps', args.checkpoint))
        assert os.path.exists(os.path.join('exps', args.checkpoint))
        checkpoint_file = os.path.join('exps', args.checkpoint, 'Model', 'checkpoint_best.pth')
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch']
        uncompressed_fid = checkpoint['best_fid']
        #best_fid = checkpoint['best_fid']  # reset best_fid, to enable saving best-fid after compression
        if 'gen' in checkpoint.keys():
            print('Loading generator from checkpoint ...')
            gen_net = checkpoint['gen']
            if isinstance(gen_net, torch.nn.DataParallel):
                gen_net = gen_net.module
            gen_net = torch.nn.DataParallel(gen_net, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])
            gen_net.load_state_dict(checkpoint['gen_state_dict'])
        else:
            print('Loading generator state_dict from checkpoint ...')
            gen_net.load_state_dict(checkpoint['gen_state_dict'])
        if 'dis' in checkpoint.keys():
            print('Loading discriminator from checkpoint ...')
            dis_net = checkpoint['dis']
            if isinstance(dis_net, torch.nn.DataParallel):
                dis_net = dis_net.module
            dis_net = torch.nn.DataParallel(dis_net, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])
            dis_net.load_state_dict(checkpoint['dis_state_dict'])
        else:
            print('Loading discriminator state_dict from checkpoint ...')
            dis_net.load_state_dict(checkpoint['dis_state_dict'])
        
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        
        avg_gen_net = deepcopy(gen_net)
        avg_gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        gen_avg_param = copy_params(avg_gen_net)
        del avg_gen_net

        args.path_helper = set_root_dir('exps', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])
        logger.info(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
    else:
        args.path_helper = set_root_dir('exps', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])
        print('Continuing with random initialinzed model for degbuging.')
        #raise NotImplementedError('no checkpoint is given')

    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }

    # model size
    initial_gen_size = count_parameters_in_MB(gen_net)
    initial_gen_paramsize = sum(p.numel() for p in gen_net.parameters())
    logger.info('Initial Param size of G = %fMB', initial_gen_size)
    logger.info('Initial Param size of D = %fMB', count_parameters_in_MB(dis_net))
    logger.info('Initial # of params in G = %d', initial_gen_paramsize)
    flops0 = print_FLOPs(basemodel_gen, (1, args.latent_dim), logger)
    print_FLOPs(basemodel_dis, (1, 3, args.img_size, args.img_size), logger)
    
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (100, args.latent_dim)))

    improvement_count = 6
    icounter = improvement_count
    logger.info(f'Upper bound: {args.bu} and Lower Bound: {args.bl}.')
    logger.info(f'Best FID score: {best_fid}. Best IS score: {best_is}.')

    # Compression loop
    steps = len(args.layers)
    replaced_layers = {}
    epoch=0
    gp=0

    performance_dict = {'fid':[], 'is':[], 'e':[]}
    compression_dict = {'layers':[], 'ranks':[], 'gen_size':[], 'flops':[], 'layer_reduction_ratio':[], 'total_reduction_ratio':[], 'init_flops': flops0, 'init_gen_size': initial_gen_size}

    # Evaluate Before compression
    backup_param = copy_params(gen_net)
    load_params(gen_net, gen_avg_param)
    inception_score, std, fid_score = validate(args, fixed_z, fid_stat, gen_net, writer_dict)
    logger.info(f'Initial Inception score mean: {inception_score}, Inception score std: {std}, '
                f'FID score: {fid_score} || @ epoch {epoch}.')
    load_params(gen_net, backup_param)

    performance_dict['fid'].append(fid_score)
    performance_dict['is'].append(inception_score)

    
    logger.info('Layers to be compressed: %s', args.layers)
    if args.byratio:
        args.rank = get_ranks_per_layer(gen_net.module, args.compress_ratio, args.layers)
        logger.info('Ranks per layer for compress-ratio of %f: %s', args.compress_ratio, args.rank)
    else:
        logger.info('Ranks per layer for given ranks: %s', args.rank)

    e_=0
    for step in range(steps):

        step_path_dict = set_step_dir(args.path_helper, step, args.layers[step], args.rank[step])
        layer_name = args.layers[step]
        rank = args.rank[step]
        logger.info('Decomposing layer {}, with cp rank {}.'.format(layer_name, rank))

        #get the index of the layer, in gen_avg_param list, to be replaced after decomposition
        indx = 0
        for gen_avg_param_i, (name, param) in zip(gen_avg_param, gen_net.named_parameters()):
            #print(name)
            if name == ('module.'+layer_name+'.weight'):
                print('found at index ', indx)
                assert(gen_avg_param_i.shape == param.shape)
                break
            else:
                indx += 1

        new_layers, approx_error, layer_compress_ratio, decomp_rank = decompose_and_replace_conv_layer_by_name(gen_net.module, layer_name, rank=rank, freeze=False, device=args.gpu_ids[0])

        replaced_layers[layer_name] = {'rank':rank, 'approx_error':approx_error, 'layer_compress_ratio': layer_compress_ratio}
        torch.save({'layers':args.layers, 'steps_completed':step, 'ranks':args.rank,  'replaced_layers':replaced_layers}, os.path.join(step_path_dict['prefix'],'decompose_info.pth'))

        logger.info('Layer Approximation error: {}, Layer Reduction ratio: {}'.format(approx_error[-1], layer_compress_ratio))

        step_gen_size = count_parameters_in_MB(gen_net)
        step_flops = print_FLOPs(gen_net, (1, args.latent_dim), logger)
        
        step_gen_paramsize = sum(p.numel() for p in gen_net.parameters())
        logger.info('Param size of G at step %d = %fMB', step, step_gen_size)
        logger.info('Number of parameters of G at step %d = %d', step, step_gen_paramsize)

        logger.info(f'Total number of parameters: {initial_gen_paramsize}. Total number of parameters is reduced by {initial_gen_paramsize-step_gen_paramsize} ({((initial_gen_paramsize-step_gen_paramsize)/initial_gen_paramsize)*100:.2f}%)')
        compression_dict['layers'].append(layer_name)
        compression_dict['ranks'].append(decomp_rank)
        compression_dict['gen_size'].append(step_gen_size)
        compression_dict['flops'].append(step_flops)
        compression_dict['layer_reduction_ratio'].append(layer_compress_ratio)
        compression_dict['total_reduction_ratio'].append(((initial_gen_paramsize-step_gen_paramsize)/initial_gen_paramsize)*100)
        torch.save(compression_dict, os.path.join(args.path_helper['prefix'],'compression_info.pth'))



        gen_net = gen_net.cuda(args.gpu_ids[0])

        logger.info('Generator modules after step %d compression: %s', step, [n for n, p in gen_net.named_parameters()])
        print('assert( %d = %d )',len(gen_avg_param), len(list(gen_net.parameters())))


        # The gen_avg_param of the compressed layer must be replaced with the new compressed layer
        gen_avg_param.pop(indx)
        for n, p in new_layers.named_parameters():#gen_net.named_parameters():
            #if layer_name in n and 'weight' in n:
            if 'weight' in n:
                gen_avg_param.insert(indx, deepcopy(p.detach()))
                print(p.shape)
                indx += 1
        print(len(gen_avg_param), len(list(gen_net.parameters())))


        # Evaluate after compresion step
        backup_param = copy_params(gen_net)
        load_params(gen_net, gen_avg_param)
        inception_score, std, fid_score = validate(args, fixed_z, fid_stat, gen_net, writer_dict)
        #inception_score, std, fid_score = 0,0,0
        logger.info(f'(Post-compression, @ step {step} ) Inception score mean: {inception_score}, Inception score std: {std}, '
                    f'FID score: {fid_score} || @ epoch {epoch}.')
        load_params(gen_net, backup_param)

        performance_dict['fid'].append(fid_score)
        performance_dict['is'].append(inception_score)
        performance_dict['e'].append(e_)
        e_+=1

        if (args.compress_mode == 'allatonce') and (step < (steps-1)):
            logger.info('plotting performance for step {}'.format(step))
            plot_performance(performance_dict, step_path_dict['prefix'])
            logger.info('skipping fine-tunine for step {}'.format(step))
            continue
        elif (args.compress_mode =='grouped') and  (step < (sum(args.groups[:gp+1]) -1)) :
            logger.info('plotting performance for step {}'.format(step))
            plot_performance(performance_dict, step_path_dict['prefix'])
            logger.info('skipping fine-tunine for step {}'.format(step))
            continue
        elif (args.compress_mode =='grouped') and  (step == (sum(args.groups[:gp+1]) - 1)):
            gp+=1
            #plot_performance(performance_dict, step_path_dict['prefix'])
            step_epochs = args.ft_epochs
            logger.info('Grouped decomposition: Group {}, gp-epochs: {}'.format(gp, step_epochs))
        else:   #sequential
            step_epochs = args.ft_epochs
            logger.info('plotting performance for step {}'.format(step))
            plot_performance(performance_dict, step_path_dict['prefix'])


        args.max_epoch_D=step_epochs
        # train loop
        for epoch in tqdm(range(int(0), int(step_epochs)), desc='total progress'):
            lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
            train(args, gen_net, dis_net, gen_optimizer, dis_optimizer,
                gen_avg_param, train_loader, epoch, writer_dict, lr_schedulers)

            if epoch % args.val_freq == 0 or epoch == int(step_epochs)-1:
                backup_param = copy_params(gen_net)
                load_params(gen_net, gen_avg_param)
                inception_score, std, fid_score = validate(args, fixed_z, fid_stat, gen_net, writer_dict)
                logger.info(f'(FT @ compression step {step}) Inception score mean: {inception_score}, Inception score std: {std}, '
                            f'FID score: {fid_score} || @ epoch {epoch}.')
                performance_dict['fid'].append(fid_score)
                performance_dict['is'].append(inception_score)
                performance_dict['e'].append(e_)
                load_params(gen_net, backup_param)
                plot_performance(performance_dict, step_path_dict['prefix'])
                if fid_score < best_fid:
                    best_fid = fid_score
                    is_best = True
                    best_is = inception_score
                    icounter = improvement_count
                else:
                    is_best = False
                    icounter = icounter - 1
            else:
                is_best = False
            e_+=1
            
            # save model
            avg_gen_net = deepcopy(gen_net)
            load_params(avg_gen_net, gen_avg_param)
            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.arch,
                'gen_state_dict': gen_net.state_dict(),
                'dis_state_dict': dis_net.state_dict(),
                'avg_gen_state_dict': avg_gen_net.state_dict(),
                'gen_optimizer': gen_optimizer.state_dict(),
                'dis_optimizer': dis_optimizer.state_dict(),
                'best_fid': best_fid,
                'best_is' : best_is,
                'path_helper': args.path_helper,
                'gen': gen_net.module,
                'dis': dis_net.module,
                'fixed_z': fixed_z,
            }, is_best, args.path_helper['ckpt_path']) #step_path_dict['ckpt_path'])
            del avg_gen_net


            # If there is no improvement for 30 epoches then load the best model
        
            if icounter == 0:
                print(f'=> resuming from {args.path_helper["ckpt_path"]}')
                checkpoint_file = os.path.join(args.path_helper['ckpt_path'],'checkpoint_best.pth')
                assert os.path.exists(checkpoint_file)
                checkpoint = torch.load(checkpoint_file)
                start_epoch = checkpoint['epoch']
                best_fid = checkpoint['best_fid']
                best_is = checkpoint['best_is']
                gen_net.load_state_dict(checkpoint['gen_state_dict'])
                dis_net.load_state_dict(checkpoint['dis_state_dict'])
                gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
                dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
                avg_gen_net = deepcopy(gen_net)
                avg_gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
                gen_avg_param = copy_params(avg_gen_net)
                del avg_gen_net
                print(f'Upper bound changed from {args.bu} to {args.bu*2}.')
                args.bu = args.bu * 2
                logger.info(f'Upper bound: {args.bu} and Lower Bound: {args.bl}.')
                icounter = improvement_count
                logger.info(f'Best FID score: {best_fid}. Best IS score: {best_is}.')
            
        plot_performance(performance_dict, step_path_dict['prefix'])
        


if __name__ == '__main__':
    main()
