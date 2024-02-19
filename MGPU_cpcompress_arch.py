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
    validate_args(args)
    
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


    # model size
    gen_params0 = count_parameters_in_MB(gen_net)
    dis_params0 = count_parameters_in_MB(dis_net)
    gen_flops0 = print_FLOPs(basemodel_gen, (1, args.latent_dim))
    dis_flops0 = print_FLOPs(basemodel_dis, (1, 3, args.img_size, args.img_size))
    # Instanciate the Compression object
    compress_obj = Compression(gen_params0, gen_flops0)

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
        dis_net.load_state_dict(checkpoint['dis_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
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
    else:
        # create new log dir
        assert args.exp_name
        args.path_helper = set_log_dir('exps', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])

    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }

    logger.info('Initial Param size of G = %fM', gen_params0)
    logger.info('Initial Param size of D = %fM', count_parameters_in_MB(dis_net))
    logger.info('Initial FLOPs of G = %fM', gen_flops0)
    logger.info('Initial FLOPs of D = %fM', print_FLOPs(basemodel_dis, (1, 3, args.img_size, args.img_size)))

    if performance_store is None:
        performance_store = PerformanceStore()
    
    # for visualization
    if args.draw_arch:
        from utils.genotype import draw_graph_G, draw_graph_D
        draw_graph_G(genotype_G, save=True, file_path=os.path.join(args.path_helper['graph_vis_path'], 'latest_G'))
        # draw_graph_D(genotype_D, save=True, file_path=os.path.join(args.path_helper['graph_vis_path'], 'latest_D'))

    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (100, args.latent_dim)))

    # Evaluate before compression
    if args.eval_before_compression:
        # Evaluate Before compression
        backup_param = copy_params(gen_net)
        load_params(gen_net, gen_avg_param)
        inception_score, std, fid_score = validate(args, fixed_z, fid_stat, gen_net, writer_dict)
        logger.info(f'Initial Inception score mean: {inception_score}, Inception score std: {std}, '
                    f'FID score: {fid_score} || @ epoch {start_epoch}.')
        load_params(gen_net, backup_param)
        performance_store.set_init(fid_score, inception_score)
        performance_store.plot(args.path_helper['prefix'])

    # Apply compression on all layers of the model (one-shot)
    logger.info(f'args.layers:{args.layers}')
    removed_params = {}
    for name, param in gen_net.named_parameters():
        logger.info(f'scanning for:{name}')
        if any([name[:len('module.'+layer)]=='module.'+layer for layer in args.layers]):
            logger.info(f'found:{name}')
            removed_params[name]=param
    logger.info(f'Removed params:{removed_params.keys()}')

    gen_avg_param, compression_info, decomposition_info = compress_obj.apply_compression(args, gen_net, gen_avg_param, args.layers, args.rank, logger)

    if args.freeze_before_compressed:
        logger.info(f'freezing the layers before {args.layers[0]}...')
        assert(len(args.layers) == 1)
        for name, param in gen_net.named_parameters():
            if args.layers[0] in name:
                break
            param.requires_grad = False

    elif args.reverse_g_freeze:
        for param in gen_net.parameters():
            param.requires_grad = not param.requires_grad

    for name, param in gen_net.named_parameters():
        logger.info(f"{name}-{param.requires_grad}")
    logger.info('------------------------------------------')
    for name, param in dis_net.named_parameters():
        logger.info(f"{name}-{param.requires_grad}")

    # Evaluate after compression
    logger.info('------------------------------------------')
    logger.info('Performance Evaluation After compression')
    backup_param = copy_params(gen_net)
    load_params(gen_net, gen_avg_param)
    
    inception_score, std, fid_score = validate(args, fixed_z, fid_stat, gen_net, writer_dict)
    logger.info(f'Inception score mean: {inception_score}, Inception score std: {std}, '
                f'FID score: {fid_score} || after compression.')
    load_params(gen_net, backup_param)
    performance_store.update(fid_score, inception_score, start_epoch)
    performance_store.plot(args.path_helper['prefix'])
    
    # set optimizer after compression
    #gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
    #                                 args.g_lr, (args.beta1, args.beta2))
    #dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
    #                                 args.d_lr, (args.beta1, args.beta2))
    old_params = []
    old_param_names = []
    new_params = []
    new_param_names = []
    for name, param in gen_net.named_parameters():
        if param in gen_optimizer.state.keys():
            old_params.append(param)
            old_param_names.append(name)
        else:
            new_params.append(param)
            new_param_names.append(name)
    logger.info(f'old_params: {old_param_names}')
    logger.info(f'new_params: {new_param_names}')

    new_gen_optimizer = torch.optim.Adam(old_params, args.g_lr, (args.beta1, args.beta2))
    new_gen_optimizer.add_param_group({'params': new_params, 'lr': 1e-8, 'betas': (args.beta1, args.beta2)})
    new_dis_optimizer = torch.optim.Adam(dis_net.parameters(), args.d_lr, (args.beta1, args.beta2))
    for name, param in gen_net.named_parameters():
        if param in gen_optimizer.state.keys():
            new_gen_optimizer.state[param] = gen_optimizer.state[param]
            new_gen_optimizer.state[param]['exp_avg'] = gen_optimizer.state[param]['exp_avg'].clone()
            new_gen_optimizer.state[param]['exp_avg_sq'] = gen_optimizer.state[param]['exp_avg_sq'].clone()
            print(new_gen_optimizer.state[param]['exp_avg'].shape, param.shape)
        else:
            if 'bias' in name:
                name2 = name.rsplit('.',1)[0].rsplit('.',1)[0]+'.'+name.rsplit('.',1)[1]
                for n_, p_ in removed_params.items():
                    if n_ == name2:
                        new_gen_optimizer.state[param] = gen_optimizer.state[p_]
                        new_gen_optimizer.state[param]['exp_avg'] = gen_optimizer.state[p_]['exp_avg'].clone()
                        new_gen_optimizer.state[param]['exp_avg_sq'] = gen_optimizer.state[p_]['exp_avg_sq'].clone()
                        print(new_gen_optimizer.state[param]['exp_avg'].shape, param.shape)

        print(name, param in gen_optimizer.state.keys())

    #gen_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, gen_net.parameters()),
    #                                 args.g_lr, momentum=0.9)
    #dis_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, dis_net.parameters()),
    #                                 args.d_lr, momentum=0.9)
    
    epoch = 0
    best_fid = fid_score
    best_is = inception_score
    is_best = True
    # save the model right after compression
    logger.info('------------------------------------------')
    logger.info('Saving the model After compression')
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
        'best_is': best_is,
        'path_helper': args.path_helper,
        'compression_info': compression_info,
        'decomposition_info': decomposition_info,
        'performance_store': performance_store,
    }, is_best, args.path_helper['ckpt_path'])
    del avg_gen_net
    logger.info('------------------------------------------')
    logger.info(f"Saving the model at {args.path_helper['ckpt_path']}")

    # train loop
    for epoch in tqdm(range(int(start_epoch), int(args.max_epoch_D)), desc='total progress'):
        lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
        train(args, gen_net, dis_net, new_gen_optimizer, new_dis_optimizer,
              gen_avg_param, train_loader, epoch, writer_dict, lr_schedulers)

        if epoch % args.val_freq == 0 or epoch == int(args.max_epoch_D)-1:
            backup_param = copy_params(gen_net)
            load_params(gen_net, gen_avg_param)
            inception_score, std, fid_score = validate(args, fixed_z, fid_stat, gen_net, writer_dict)
            logger.info(f'Inception score mean: {inception_score}, Inception score std: {std}, '
                        f'FID score: {fid_score} || @ epoch {epoch}. || Best FID score: {best_fid}')
            load_params(gen_net, backup_param)
            performance_store.update(fid_score, inception_score, epoch)
            performance_store.plot(args.path_helper['prefix'])
            if fid_score < best_fid:
                best_fid = fid_score
                best_is = inception_score
                is_best = True
            else:
                is_best = False
        else:
            is_best = False
        
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
            'best_is': best_is,
            'path_helper': args.path_helper,
            'compression_info': compression_info,
            'decomposition_info': decomposition_info,
            'performance_store': performance_store,
        }, is_best, args.path_helper['ckpt_path'])
        del avg_gen_net


if __name__ == '__main__':
    main()
