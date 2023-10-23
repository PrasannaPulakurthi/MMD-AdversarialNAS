import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.random import random_cp
from tensorly.decomposition import tucker, partial_tucker
from tensorly.decomposition._tucker import initialize_tucker
import matplotlib.pyplot as plt
import os
import logging
from utils.utils import count_parameters_in_MB
from utils.flop_benchmark import print_FLOPs
from copy import deepcopy

tl.set_backend('pytorch')

_logger = logging.getLogger(__name__)

def get_conv2d_layers_info(model):

    conv_layers_info = {}
    for name, l in model.named_modules():
        if isinstance(l, nn.Conv2d):
            conv_layers_info[name] = l.weight.shape

    return conv_layers_info
    
def decompose_and_replace_conv_layers(module, replaced_layers, rank=None, device='cpu'):
    
    if rank is None:
        raise ValueError("Please specify a rank for decomposition")
    
    rank = torch.tensor(rank, dtype=torch.int32)
    if device=='cuda':
        rank=rank.to('cuda' if device=='cuda' else 'cpu')
    
    # decompose convolutional layers of a given module using CP decomposition
    
    error = None
    for name, layer in module.named_children():
        if isinstance(layer, nn.Conv2d):
            layer_key = f"{name}_{id(layer)}"
            if layer_key not in replaced_layers:
                print(name)
                replaced_layers[layer_key] = layer
                new_layers, error, layer_compress_ratio = cp_decomposition_con_layer(layer, rank)
                new_layers = new_layers.to(device)
                setattr(module, name, new_layers)
                break
        else:
            continue
    return replaced_layers, error

def decompose_and_replace_conv_layer_by_name(module, layer_name, rank=None, freeze=False, device='cpu', decomposition='cp', replace_only=False):    # rank must be int/tuple/list for tucker
    
    if rank is None:
        raise ValueError("Please specify a rank for decomposition")
    
    rank = torch.tensor(rank, dtype=torch.int32)
    #if device=='cuda':
    #    rank=rank.to('cuda' if device=='cuda' else 'cpu')

    
    # decompose convolutional layers of a given module using CP decomposition
    error = None
    queue = [(name,layer,module,name) for name, layer in list(module.named_children())]
    while queue:
        (name,layer,parent,fullname) = queue.pop()
        if isinstance(layer,nn.Conv2d):
            if layer_name == fullname:
                if decomposition == 'cp':
                    new_layers, error, layer_compress_ratio, rank = cp_decomposition_con_layer(layer, rank, replace_only=replace_only)
                #elif decomposition == 'tucker':
                #    new_layers, error, layer_compress_ratio, rank = tucker_decompose_con_layer(layer, rank)
                else:
                    raise('Unknown decomposition method.')
                new_layers = new_layers.to(device)
                setattr(parent, name, new_layers)
                break
        elif isinstance(layer,nn.Linear) and layer_name == fullname:
            new_layers, error, layer_compress_ratio, rank = cp_decomposition_fc_layer(layer, rank, replace_only=replace_only)
            new_layers = new_layers.to(device)
            setattr(parent, name, new_layers)
            break
        
        children = list(layer.named_children())
        if len(children)>0:
            queue.extend([(name,child,layer,fullname+'.'+name) for name,child in children])
    
    if freeze:  # freeze just the given layer
        for name, param in new_layers.named_parameters():
                param.requires_grad = False
            #param.requires_grad = False
    return  new_layers, error, layer_compress_ratio, rank

def cp_decomposition_fc_layer(layer, rank, replace_only=False):

    layer_total_params = sum(p.numel() for p in layer.parameters()) ##newline
    if not replace_only:
        cont = True
        while cont:
            #(weights, factors), decomp_err = parafac(layer.weight.data, rank=rank, init='random', return_errors=True)
            (weights, factors), decomp_err = parafac(layer.weight.data, rank=rank, init='random', return_errors=True, normalize_factors=True, orthogonalise=True)
            print('cp weights (must be 1): ', weights)
            const = torch.sqrt(torch.sqrt(weights)) #added to distribute the weights equally
            factors[0] = factors[0]*const #added to distribute the weights equally
            factors[1] = factors[1]*const #added to distribute the weights equally
            weights = torch.ones(rank) #added to distribute the weights equally
            #decomp_err.append(torch.norm(tl.cp_tensor.cp_to_tensor((weights, factors))-layer.weight.data)/torch.norm(layer.weight.data)) #added to distribute the weights equally
            c_out, c_in = factors[0], factors[1]
            if torch.isnan(c_out).any() or torch.isnan(c_in).any():
                _logger.info(f"NaN detected in CP decomposition, trying again with rank {int(rank/2)}")
                rank = int(rank/2)
            else:
                cont = False
    else:
        (_,factors) = random_cp(layer.weight.data.shape, rank=rank)
        c_out, c_in = factors[0], factors[1]
        decomp_err = None

    bias_flag = layer.bias is not None

    fc_1 = torch.nn.Linear(in_features=c_in.shape[0], \
            out_features=rank, bias=False)

    fc_2 = torch.nn.Linear(in_features=rank, 
            out_features=c_out.shape[0], bias=bias_flag)

    if bias_flag:
        fc_2.bias.data = layer.bias.data
    
    fc_1.weight.data = torch.transpose(c_in,1,0)
    fc_2.weight.data = c_out

    new_layers = nn.Sequential(fc_1, fc_2)

    layer_compressed_params= sum(p.numel() for p in new_layers.parameters()) ##newline
    layer_compress_ratio = ((layer_total_params-layer_compressed_params)/layer_total_params)*100 ##newline

    return new_layers, decomp_err, layer_compress_ratio, rank

def cp_decomposition_con_layer(layer, rank, replace_only=False):
    stride0 = layer.stride[0]
    stride1 = layer.stride[1]
    padding0 = layer.padding[0]
    padding1 = layer.padding[1]

    layer_total_params = sum(p.numel() for p in layer.parameters()) ##newline
    if not replace_only:
        cont = True
        while cont:
            #(weights, factors), decomp_err = parafac(layer.weight.data, rank=rank, init='random', return_errors=True)
            (weights, factors), decomp_err = parafac(layer.weight.data, rank=rank, init='random', return_errors=True, normalize_factors=True, orthogonalise=True)
            const = torch.sqrt(torch.sqrt(weights)) #added to distribute the weights equally
            factors[0] = factors[0]*const #added to distribute the weights equally
            factors[1] = factors[1]*const #added to distribute the weights equally
            factors[2] = factors[2]*const #added to distribute the weights equally
            factors[3] = factors[3]*const #added to distribute the weights equally
            weights = torch.ones(rank) #added to distribute the weights equally
            #decomp_err.append(torch.norm(tl.cp_tensor.cp_to_tensor((weights, factors))-layer.weight.data)/torch.norm(layer.weight.data)) #added to distribute the weights equally
            c_out, c_in, x, y = factors[0], factors[1], factors[2], factors[3]
            if torch.isnan(c_out).any() or torch.isnan(c_in).any() or torch.isnan(x).any() or torch.isnan(y).any():
                _logger.info(f"NaN detected in CP decomposition, trying again with rank {int(rank/2)}")
                rank = int(rank/2)
            else:
                cont = False
    else:
        (_,factors) = random_cp(layer.weight.data.shape, rank=rank)
        c_out, c_in, x, y = factors[0], factors[1], factors[2], factors[3]
        decomp_err = None

    bias_flag = layer.bias is not None

    pointwise_s_to_r_layer = torch.nn.Conv2d(in_channels=c_in.shape[0], \
            out_channels=rank, kernel_size=1, stride=1, padding=0, 
            dilation=layer.dilation, bias=False)

    depthwise_vertical_layer = torch.nn.Conv2d(in_channels=rank, 
            out_channels=rank, kernel_size=(x.shape[0], 1),
            stride=1, padding=(layer.padding[0], 0), dilation=layer.dilation,
            groups=rank, bias=False)

    depthwise_horizontal_layer = \
        torch.nn.Conv2d(in_channels=rank, \
            out_channels=rank, 
            kernel_size=(1, y.shape[0]), stride=layer.stride,
            padding=(0, layer.padding[0]), 
            dilation=layer.dilation, groups=rank, bias=False)

    pointwise_r_to_t_layer = torch.nn.Conv2d(in_channels=rank, \
            out_channels=c_out.shape[0], kernel_size=1, stride=1,
            padding=0, dilation=layer.dilation, bias=bias_flag)
    if bias_flag:
        pointwise_r_to_t_layer.bias.data = layer.bias.data
    #pointwise_r_to_t_layer.bias.data = layer.bias.data
    depthwise_horizontal_layer.weight.data = \
        torch.transpose(y, 1, 0).unsqueeze(1).unsqueeze(1)
    depthwise_vertical_layer.weight.data = \
        torch.transpose(x, 1, 0).unsqueeze(1).unsqueeze(-1)
    pointwise_s_to_r_layer.weight.data = \
        torch.transpose(c_in, 1, 0).unsqueeze(-1).unsqueeze(-1)
    pointwise_r_to_t_layer.weight.data = c_out.unsqueeze(-1).unsqueeze(-1)

    new_layers = nn.Sequential(pointwise_s_to_r_layer, depthwise_vertical_layer, \
                    depthwise_horizontal_layer, pointwise_r_to_t_layer)

    layer_compressed_params= sum(p.numel() for p in new_layers.parameters()) ##newline 
    layer_compress_ratio = ((layer_total_params-layer_compressed_params)/layer_total_params)*100 ##newline

    return new_layers, decomp_err, layer_compress_ratio, rank

def tucker_decompose_con_layer(layer, rank):
    stride0 = layer.stride[0]
    stride1 = layer.stride[1]
    padding0 = layer.padding[0]
    padding1 = layer.padding[1]

    try:
        rank0 = rank[0]
        rank1 = rank[1]
    except:
        if len(rank.shape) == 0:
            rank0 = rank
            rank1 = rank
        else:
            rank0 = rank[0]
            rank1 = rank[0]
    layer_total_params = sum(p.numel() for p in layer.parameters()) ##newline
    cont = True
    while cont:
        print(layer.weight.data.shape)
        init = init_tucker_with_eye_spatial_modes(layer.weight.data, rank=[rank1, rank0, layer.weight.data.shape[2],layer.weight.data.shape[3]], init='random')
        print(init[0].shape)
        for f in init[1]:
            print(f.shape)
        (weights, factors), decomp_err = partial_tucker(layer.weight.data, 
                                                        rank=[rank1, rank0, layer.weight.data.shape[2],layer.weight.data.shape[3]],
                                                        init=init, modes=[0,1])
        c_out, c_in, x, y = factors[0], factors[1], factors[2], factors[3]
        _logger.info(f"factors 2,3 must be eye matrices", factors[2], factors[3])
        if torch.isnan(c_out).any() or torch.isnan(c_in).any() or torch.isnan(x).any() or torch.isnan(y).any():
            _logger.info(f"NaN detected in Tucker decomposition, trying again with rank {int(rank0/2, rank1/2)}")
            rank0 = min(1, int(rank0/2))
            rank1 = min(1, int(rank1/2))
        else:
            cont = False

    bias_flag = layer.bias is not None

    pointwise_s_to_r_layer = torch.nn.Conv2d(in_channels=c_in.shape[0], \
            out_channels=rank0, kernel_size=1, stride=1, padding=0, 
            dilation=layer.dilation, bias=False)

    spatial_layer = torch.nn.Conv2d(in_channels=rank0, 
            out_channels=rank1, kernel_size=(x.shape[0], y.shape[0]),
            stride=1, padding=(layer.padding[0], layer.padding[1]), dilation=layer.dilation, bias=False)

    pointwise_r_to_t_layer = torch.nn.Conv2d(in_channels=rank1, \
            out_channels=c_out.shape[0], kernel_size=1, stride=1,
            padding=0, dilation=layer.dilation, bias=bias_flag)
    if bias_flag:
        pointwise_r_to_t_layer.bias.data = layer.bias.data

    pointwise_s_to_r_layer.weight.data = \
        torch.transpose(c_in, 1, 0).unsqueeze(-1).unsqueeze(-1)
    spatial_layer.weight.data = weights
    pointwise_r_to_t_layer.weight.data = c_out.unsqueeze(-1).unsqueeze(-1)

    new_layers = nn.Sequential(pointwise_s_to_r_layer, spatial_layer, pointwise_r_to_t_layer)

    layer_compressed_params= sum(p.numel() for p in new_layers.parameters()) ##newline
    layer_compress_ratio = ((layer_total_params-layer_compressed_params)/layer_total_params)*100 ##newline

    return new_layers, decomp_err, layer_compress_ratio, (rank0, rank1)
    
def get_conv2d_layer_approximation_vs_rank(model, conv_layer_name, cp_ranks=None, max_rank = None, decompose_type='cp', save_fig=False, save_path=None):
    
    for name, l in model.named_modules():
        if name == conv_layer_name:
            layer = l
            break
    #layer = model._modules[conv_layer_name]
    W = layer.weight.data.cpu()
    w_size = W.shape

    if not decompose_type == 'cp':
        raise('Not implemented yet')
    if cp_ranks is None:
        cp_ranks   = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    # cp decomposition
    if max_rank is None:
        max_rank = min(w_size[0], w_size[1])
    approximations = []
    ranks = []

    for rank in cp_ranks:
        
        if max_rank and rank > max_rank:
            break
        print('Rank: {}'.format(rank))
        (weights, factors), decomp_err = parafac(W, rank=rank, init='random', return_errors=True)
        approx_error = decomp_err[-1]
        approximations.append(approx_error)
        ranks.append(rank)

    if save_fig and save_path is not None:
        f = plt.figure()
        plt.plot(ranks, approximations)
        plt.xlabel('Rank')
        plt.ylabel('Kernel Approximation error')
        plt.title('Layer: {}'.format(conv_layer_name))
        plt.savefig(os.path.join(save_path, '{}_approximation_error_vs_rank.png'.format(conv_layer_name)))
        plt.close(f)

    return ranks, approximations

def init_tucker_with_eye_spatial_modes(tensor, rank, init='random'):
    core, factors = initialize_tucker(
        tensor,
        rank,
        random_state=None,
        modes=[0,1],#,2,3],
        init=init,
    )
    # set the last two factors to identity
    factors[2] = torch.eye(factors[1].shape[0])
    factors[3] = torch.eye(factors[2].shape[0])
    for i in range(len(factors)):
        factors[i] = factors[i].to("cuda")
    core = core.to("cuda")

    return (core, factors)


class DecompositionInfo:
    def __init__(self):
        self.layers = []
        self.ranks = []
        self.approx_error = []

    def append(self, layer, rank, approx_error):
        self.layers.append(layer)
        self.ranks.append(rank)
        self.approx_error.append(approx_error)        
        
class CompressionInfo:
    def __init__(self, initial_size=None, initial_flops=None):
        self.layers = []
        self.ranks = []
        self.initial_size = initial_size
        self.initial_flops = initial_flops
        self.sizes = []
        self.flops = []
        self.per_layer_reduction_ratio = []     # has same order as layers, ranks
        
        self.total_size_reduction_ratio = []
        self.total_flops_reduction_ratio = []
    
    def add(self, layer, rank, size, flops, layer_reduction_ratio):
        self.layers.append(layer)
        self.ranks.append(rank)
        self.sizes.append(size)
        self.flops.append(flops)
        
        self.per_layer_reduction_ratio.append(layer_reduction_ratio)

        self.total_size_reduction_ratio.append(100*(self.initial_size - size)/self.initial_size)
        self.total_flops_reduction_ratio.append(100*(self.initial_flops - flops)/self.initial_flops)
    
    def get_compression_ratio(self):
        try:
            return self.total_size_reduction_ratio[-1]
        except:
            print('No compression ratio found')
            return 0

class Compression:
    def __init__(self, size0, flops0):
        self.init_size = size0
        self.init_flops = flops0
        self.decomposition_info = DecompositionInfo()
        self.compression_info = CompressionInfo(size0, flops0)

    def apply_decomposition_from_checkpoint(self, args, network, decomposition_info:DecompositionInfo, compression_info: CompressionInfo = None, replace_only=False): ##TODO
        for layer, rank in zip(decomposition_info.layers, decomposition_info.ranks):
            self.apply_layer_compression(args, network, layer, rank, replace_only=replace_only)
        self.decomposition_info = decomposition_info
        if compression_info is not None:
            self.compression_info = compression_info


    def apply_layer_compression(self, args, network, layer, rank, logger=None, avg_param=None, replace_only=False):
        try:
            logger.info('\nDecomposing layer {} with rank {}'.format(layer, rank))
        except:
            print('\nDecomposing layer {} with rank {}'.format(layer, rank))


        if avg_param is not None:
            indx = 0
            for avg_param_i, (name, param) in zip(avg_param, network.named_parameters()):
                if name == ('module.'+layer+'.weight'):
                    print('found at index ', indx)
                    assert(avg_param_i.shape == param.shape)
                    break
                else:
                    indx += 1
        if args.freeze_layers and (layer in args.freeze_layers):
            if logger:
                logger.info('Freezing layer {}'.format(layer))
            freeze = True
        else:
            freeze = False
        new_layers, approx_error, layer_compress_ratio, decomp_rank = decompose_and_replace_conv_layer_by_name(network.module, layer, rank=rank, freeze=freeze, device=args.gpu_ids[0], replace_only=replace_only)
        # calculate sizes after layer decomposition
        step_size = count_parameters_in_MB(network)
        step_flops = 0 # print_FLOPs(network, (1, args.latent_dim), logger)

        self.compression_info.add(layer, rank, step_size, step_flops, layer_compress_ratio)
        if logger is not None:
            logger.info('Param size of G after decomposing %s = %fM',layer, step_size)
            # logger.info('FLOPs of G at step after decomposing %s = %fG', layer, step_flops)
            logger.info('Compression ratio of G at step %s  = %f', layer, self.compression_info.get_compression_ratio())
        else:
            print(f"Param size of G after decomposing {layer} = {step_size}M")
            # print(f"FLOPs of G at step after decomposing {layer} = {step_flops}M")
            print(f"Compression ratio of G at step {layer}  = {self.compression_info.get_compression_ratio()}")

        if not replace_only:  
            self.decomposition_info.append(layer=layer, rank=decomp_rank, approx_error=approx_error[-1])
            if logger is not None:
                logger.info('Layer Approximation error: {}, Layer Reduction ratio: {}'.format(approx_error[-1], layer_compress_ratio))
            else:
                print('Layer Approximation error: {}, Layer Reduction ratio: {}'.format(approx_error[-1], layer_compress_ratio))



        if avg_param is not None:
            # The gen_avg_param of the compressed layer must be replaced with the new compressed layer
            avg_param.pop(indx)
            for n, p in new_layers.named_parameters():#gen_net.named_parameters():
                #if layer_name in n and 'weight' in n:
                if 'weight' in n:
                    avg_param.insert(indx, deepcopy(p.detach()))
                    indx += 1

        return avg_param

    def apply_compression(self, args, network, avg_param, layers, ranks, logger): ##TODO
        steps = len(layers)
        for step in range(steps):
            layer_name = layers[step]
            rank = ranks[step]
            avg_param = self.apply_layer_compression(args, network, layer_name, rank, logger, avg_param)

        return avg_param, self.compression_info, self.decomposition_info

