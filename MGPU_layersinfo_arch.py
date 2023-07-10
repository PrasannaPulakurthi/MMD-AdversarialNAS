# @Date    : 2019-10-22
# @Author  : Chen Gao

import cfg
import archs
from network import validate, load_params, copy_params
from utils.utils import set_log_dir, create_logger, count_parameters_in_MB
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception
from utils.flop_benchmark import print_FLOPs

import torch
import os
import numpy as np
from tensorboardX import SummaryWriter
from copy import deepcopy

from decompose import get_conv2d_layers_info, get_conv2d_layer_approximation_vs_rank
from utils.compress_utils import get_ranks_per_layer

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)

    args.exp_name = 'layersinfo-'+args.exp_name

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
    

    # fid stat
    if args.dataset.lower() == 'cifar10':
        fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
    elif args.dataset.lower() == 'stl10':
        fid_stat = 'fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
    else:
        raise NotImplementedError(f'no fid stat for {args.dataset.lower()}')
    assert os.path.exists(fid_stat)

    # set writer
    #print(f'=> resuming from {args.checkpoint}')
    #assert os.path.exists(os.path.join('exps', args.checkpoint))
    #checkpoint_file = os.path.join('exps', args.checkpoint, 'Model', 'checkpoint_best.pth')
    #assert os.path.exists(checkpoint_file)
    #heckpoint = torch.load(checkpoint_file)
    #epoch = checkpoint['epoch'] - 1
    #gen_net.load_state_dict(checkpoint['gen_state_dict'])
    
    avg_gen_net = deepcopy(gen_net)
    #avg_gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
    gen_avg_param = copy_params(avg_gen_net)
    #del avg_gen_net
    assert args.exp_name
    args.path_helper = set_log_dir('exps', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    #logger.info(f'=> loaded checkpoint {checkpoint_file} (epoch {epoch})')
    epoch=0
    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'valid_global_steps': epoch // args.val_freq,
    }
    
    # model size
    gen_init_paramsize = count_parameters_in_MB(gen_net)
    logger.info('Param size of G = %fMB', gen_init_paramsize)
    print_FLOPs(basemodel_gen, (1, args.latent_dim), logger)
    
    # for visualization
    if args.draw_arch:
        from utils.genotype import draw_graph_G
        draw_graph_G(genotype_G, save=True, file_path=os.path.join(args.path_helper['graph_vis_path'], 'latest_G'))
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (100, args.latent_dim)))

    for name, param in gen_net.named_modules():
        print(name, isinstance(param, torch.nn.Conv2d),type(param))
    print('#############################################')
    logger.info('Getting conv2d layers size...')
    size=0
    size2=0
    size_weight = 0
    for name, m in gen_net.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            size+=count_parameters_in_MB(m)
            size2+=sum(p.numel() for p in m.parameters())
            size_weight+=m.weight.numel()
    
    logger.info(f'Param size of generator in Conv2d layers is: {size}MB ({size/gen_init_paramsize*100}%)')

    #for name, param in gen_net.named_parameters():
    #    print(name)
    #quit()
    # gather conv2d layers info
    logger.info('Gathering conv2d layers info...')
    conv2d_info = get_conv2d_layers_info(gen_net)

    logger.info(f'INFO: Generator has {len(conv2d_info.keys())} convolution layers:')
    for i, (layer_name, layer_shape) in enumerate(conv2d_info.items()):
        logger.info(f'({i}) Layer {layer_name}: {layer_shape}')

    # Get the parameter size ratio of conv2d layers in G
    total_conv_param_ratio=0
    logger.info('Getting the parameter size ratio of conv2d layers in G...')
    for layer_name, _ in conv2d_info.items():
        for n, layer in gen_net.named_modules():
            if n == layer_name:
                layer_paramsize = count_parameters_in_MB(layer)
                
                #logger.info(f'Layer {layer_name} param size = {layer_paramsize}MB')
                #logger.info(f'Layer {layer_name} param size ratio = {100*(layer_paramsize / gen_init_paramsize)}%\n')
                print(layer_name, 100*(layer_paramsize / gen_init_paramsize),'%')
                total_conv_param_ratio += layer_paramsize / gen_init_paramsize
                break
    logger.info(f'Total conv2d layers param size ratio = {100*total_conv_param_ratio}%')

    print('[',",".join(conv2d_info.keys()),']')
    for layer_name in conv2d_info.keys():
        for n, layer in gen_net.named_modules():
            if n == layer_name:
                layer_paramsize = count_parameters_in_MB(layer)
                
                print(layer_name, 100*(layer_paramsize / gen_init_paramsize),'%')

    ratios=[0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.01]
    
    for ratio in ratios:
        ranks = get_ranks_per_layer(gen_net, ratio, conv2d_info.keys())
        print(ratio, ' : ', ranks)
        


    # test
    #load_params(gen_net, gen_avg_param)
    #inception_score, std, fid_score = validate(args, fixed_z, fid_stat, gen_net, writer_dict)
    #logger.info(f'Inception score mean: {inception_score}, Inception score std: {std}, '
    #            f'FID score: {fid_score} || @ epoch {epoch}.')

if __name__ == '__main__':
    main()