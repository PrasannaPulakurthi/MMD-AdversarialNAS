import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.decomposition import tucker
import matplotlib.pyplot as plt
import os
import logging

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
                new_layers, error, layer_compress_ratio = cp_decomposition_con_layer(layer.cpu(), rank)
                new_layers = new_layers.to(device)
                setattr(module, name, new_layers)
                break
        else:
            continue
    return replaced_layers, error

def decompose_and_replace_conv_layer_by_name(module, layer_name, rank=None, freeze=False, device='cpu'):
    
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
                new_layers, error, layer_compress_ratio = cp_decomposition_con_layer(layer, rank)
                #print(layer)
                #print(new_layers)
                #print(len([p for p in new_layers.parameters()]))
                #print([nn for nn,_ in new_layers.named_children()])
                new_layers = new_layers.to(device)
                #print(len([p for p in module.parameters()]))
                #print(fullname)
                #print(name)
                setattr(parent, name, new_layers)
                #print(len([p for p in module.parameters()]))
                break
        
        children = list(layer.named_children())
        if len(children)>0:
            queue.extend([(name,child,layer,fullname+'.'+name) for name,child in children])
    if freeze:
        for name, param in module.named_parameters():
            if layer_name in name:
                break
            param.requires_grad = False
    return  new_layers,error, layer_compress_ratio

def cp_decomposition_con_layer(layer, rank):
    stride0 = layer.stride[0]
    stride1 = layer.stride[1]
    padding0 = layer.padding[0]
    padding1 = layer.padding[1]

    layer_total_params = sum(p.numel() for p in layer.parameters()) ##newline
    cont = True
    while cont:
        (weights, factors), decomp_err = parafac(layer.weight.data, rank=rank, init='random', return_errors=True)
        print('cp weights (must be 1): ', weights)
        c_out, c_in, x, y = factors[0], factors[1], factors[2], factors[3]
        if torch.isnan(c_out).any() or torch.isnan(c_in).any() or torch.isnan(x).any() or torch.isnan(y).any():
            _logger.info(f"NaN detected in CP decomposition, trying again with rank {int(rank/2)}")
            rank = int(rank/2)
        else:
            cont = False

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

    return new_layers, decomp_err, layer_compress_ratio
    
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