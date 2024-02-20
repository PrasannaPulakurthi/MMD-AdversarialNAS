import os
import numpy as np
import scipy.io
import math
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from imageio import imsave
from tqdm import tqdm
from copy import deepcopy
import logging

from utils.inception_score import get_inception_score
from utils.fid_score import calculate_fid_given_paths
from utils.genotype import alpha2genotype, beta2genotype, draw_graph_D, draw_graph_G


logger = logging.getLogger(__name__)


class MMD_loss(nn.Module):
    def __init__(self, bu = 4, bl = 1/4):
      super(MMD_loss, self).__init__()
      self.fix_sigma = 1
      self.bl = bl
      self.bu = bu
      return
  
    def phi(self,x,y):
      total0 = x.unsqueeze(0).expand(int(x.size(0)), int(x.size(0)), int(x.size(1)))
      total1 = y.unsqueeze(1).expand(int(y.size(0)), int(y.size(0)), int(y.size(1)))
      return(((total0-total1)**2).sum(2))
    
    def forward(self, source, target, type):
      M = source.size(dim=0)
      N = target.size(dim=0)
      # print(M,N)
      if M!=N:
        target = target[:M,:]
      L2_XX = self.phi(source, source)
      L2_YY = self.phi(target, target)
      L2_XY = self.phi(source, target)
      # print(source, target)
      bu = self.bu*torch.ones(L2_XX.size()).type(torch.cuda.FloatTensor)
      bl = self.bl*torch.ones(L2_YY.size()).type(torch.cuda.FloatTensor)
      alpha = (1/(2*self.fix_sigma))*torch.ones(1).type(torch.cuda.FloatTensor)
      m = M*torch.ones(1).type(torch.cuda.FloatTensor)
      if type == "critic":
        XX_u = torch.exp(-alpha*torch.min(L2_XX,bu))
        YY_l = torch.exp(-alpha*torch.max(L2_YY,bl))
        XX = (1/(m*(m-1))) * (torch.sum(XX_u) - torch.sum(torch.diagonal(XX_u, 0)))
        YY = (1/(m*(m-1))) * (torch.sum(YY_l) - torch.sum(torch.diagonal(YY_l, 0)))
        # loss_b = torch.mean(source.square()) + torch.mean(target.square())
        lossD = XX - YY # + 0.001*loss_b
        # print(XX, YY, loss_b)
        return lossD
      elif type == "gen":
        XX_u = torch.exp(-alpha*L2_XX)
        YY_u = torch.exp(-alpha*L2_YY)
        XY_l = torch.exp(-alpha*L2_XY)
        XX = (1/(m*(m-1))) * (torch.sum(XX_u) - torch.sum(torch.diagonal(XX_u, 0)))
        YY = (1/(m*(m-1))) * (torch.sum(YY_u) - torch.sum(torch.diagonal(YY_u, 0)))
        XY = torch.mean(XY_l)
        lossmmd = XX + YY - 2 * XY
        # eps = 1e-10*torch.tensor(1).type(torch.cuda.FloatTensor)
        # lossG = torch.sqrt(torch.max(lossmmd,eps))
        # print(XX, YY, XY)
        return lossmmd
      
def train(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch,
          writer_dict, lr_schedulers, architect_gen=None, architect_dis=None):
    writer = writer_dict['writer']
    gen_step = 0
    mmd_rep_loss = MMD_loss(args.bu, args.bl)
    
    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()
    
    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']

        real_imgs = imgs.type(torch.cuda.FloatTensor)


        # search arch of D
        if architect_dis:  
            real_imgs_w = real_imgs[:imgs.shape[0] // 2]
            real_imgs_arch = real_imgs[imgs.shape[0] // 2:]
            # sample noise
            search_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0] // 2, args.latent_dim)))
            if args.amending_coefficient:
                architect_dis.step(dis_net, real_imgs_arch, gen_net, search_z, real_imgs_train=real_imgs_w, train_z=z, eta=args.amending_coefficient)
            else:
                architect_dis.step(dis_net, real_imgs_arch, gen_net, search_z)
            # sample noise
            z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0] // 2, args.latent_dim)))
        else:
            real_imgs_w = real_imgs
            # sample noise
            z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))
            
        # train weights of D
        dis_optimizer.zero_grad()
        real_validity = dis_net(real_imgs_w)
        fake_imgs = gen_net(z).detach()
        assert fake_imgs.size() == real_imgs_w.size()
        fake_validity = dis_net(fake_imgs)
        # print(real_validity.size(), fake_validity.size())
        d_loss = mmd_rep_loss(real_validity, fake_validity,"critic")
        '''
        # Hinge loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        '''
        d_loss.backward()
        dis_optimizer.step()

        writer.add_scalar('d_loss', d_loss.item(), global_steps)

        # sample noise
        gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_bs, args.latent_dim)))
        # search arch of G
        if architect_gen:
            if global_steps % args.n_critic == 0:
                # sample noise
                search_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_bs, args.latent_dim)))
                if args.amending_coefficient:
                    architect_gen.step(search_z, gen_net, dis_net, train_z=gen_z, eta=args.amending_coefficient)
                else:
                    architect_gen.step(search_z, gen_net, dis_net)

        # train weights of G
        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()
            gen_imgs = gen_net(gen_z)
            real_validity = dis_net(real_imgs_w)
            fake_validity = dis_net(gen_imgs)
            '''
            # Hinge loss
            g_loss = -torch.mean(fake_validity)
            '''
            # print(real_validity.size(), fake_validity.size())
            g_loss = mmd_rep_loss(real_validity, fake_validity,"gen")
            g_loss.backward()
            gen_optimizer.step()

            # learning rate
            if lr_schedulers:
                gen_scheduler, dis_scheduler = lr_schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        #gen_current_lr = gen_optimizer.param_groups[0]['lr']
        #dis_current_lr = dis_optimizer.param_groups[0]['lr']
        #try:
            #print('lr.item(): gen %.6f dis %.6f' % (gen_current_lr, dis_current_lr))
        #    writer.add_scalar('g_lr', gen_current_lr.item(), global_steps)
        #    writer.add_scalar('d_lr', dis_current_lr.item(), global_steps)
        #except:
            #print('lr: gen %.6f dis %.6f' % (gen_current_lr, dis_current_lr))
        #    writer.add_scalar('g_lr', gen_current_lr, global_steps)
        #    writer.add_scalar('d_lr', dis_current_lr, global_steps)
        """
        for name, param in gen_net.named_parameters():
            layer, attr = os.path.splitext(name)
            attr = attr[1:]
            writer.add_histogram("gen-{}/{}".format(layer, attr), param, global_steps)
        for name, param in dis_net.named_parameters():
            layer, attr = os.path.splitext(name)
            attr = attr[1:]
            writer.add_histogram("dis-{}/{}".format(layer, attr), param, global_steps)
        for name, param in gen_net.named_parameters():
            if 'weight' in name and param.requires_grad:
                writer.add_scalar('gen-grad-norm2-weight/{}'.format(name), param.grad.norm(), global_steps)
            if 'bias' in name and param.requires_grad:
                writer.add_scalar('gen-grad-norm2-bias/{}'.format(name), param.grad.norm(), global_steps)

        for name, param in dis_net.named_parameters():
            if 'weight' in name and param.requires_grad:
                writer.add_scalar('dis-grad-norm2-weight/{}'.format(name), param.grad.norm(), global_steps)
            if 'bias' in name and param.requires_grad:
                writer.add_scalar('dis-grad-norm2-bias/{}'.format(name), param.grad.norm(), global_steps)
        """
        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            tqdm.write(
                '[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]' %
                (epoch, args.max_epoch_D, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1

        if architect_gen:
            # deriving arch of G/D during searching
            derive_freq_iter = math.floor((args.max_iter_D / args.max_epoch_D) / args.derive_per_epoch)
            if (args.derive_per_epoch > 0) and (iter_idx % derive_freq_iter == 0):
                genotype_G = alpha2genotype(gen_net.module.alphas_normal, gen_net.module.alphas_up, save=True,
                                            file_path=os.path.join(args.path_helper['genotypes_path'], str(epoch)+'_'+str(iter_idx)+'_G.npy'))
                genotype_D = beta2genotype(dis_net.module.alphas_normal, dis_net.module.alphas_down, save=True,
                                           file_path=os.path.join(args.path_helper['genotypes_path'], str(epoch)+'_'+str(iter_idx)+'_D.npy'))
                if args.draw_arch:
                    draw_graph_G(genotype_G, save=True,
                                 file_path=os.path.join(args.path_helper['graph_vis_path'], str(epoch)+'_'+str(iter_idx)+'_G'))
                    draw_graph_D(genotype_D, save=True,
                                 file_path=os.path.join(args.path_helper['graph_vis_path'], str(epoch)+'_'+str(iter_idx)+'_D'))


def validate(args, fixed_z, fid_stat, gen_net: nn.Module, writer_dict):
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']

    # eval mode
    gen_net = gen_net.eval()

    # generate images
    sample_imgs = gen_net(fixed_z)
    img_grid = make_grid(sample_imgs, nrow=10, normalize=True, scale_each=True)
    file_name = os.path.join(args.path_helper['sample_path'], 'img_grid.png')
    imsave(file_name, img_grid.mul_(255).clamp_(0.0, 255.0).permute(1, 2, 0).to('cpu', torch.uint8).numpy())

    # get fid and inception score
    fid_buffer_dir = os.path.join(args.path_helper['sample_path'], 'fid_buffer')
    os.makedirs(fid_buffer_dir, exist_ok=True)

    eval_iter = args.num_eval_imgs // args.eval_batch_size
    img_list = list()
    Z = []
    for iter_idx in tqdm(range(eval_iter), desc='sample images'):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))
        Z.append(z.to('cpu').numpy())
        # generate a batch of images
        gen_imgs = gen_net(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        for img_idx, img in enumerate(gen_imgs):
            file_name = os.path.join(fid_buffer_dir, f'iter{iter_idx}_b{img_idx}.png')
            imsave(file_name, img)
        img_list.extend(list(gen_imgs))
    Z = np.concatenate(Z, 0)
    # scipy.io.savemat('test_noise.mat', {'Noise': Z})

    # get inception score
    logger.info('=> calculate inception score')
    mean, std = get_inception_score(img_list)

    # get fid score
    logger.info('=> calculate fid score')
    fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None)
    
    # del buffer
    os.system('rm -r {}'.format(fid_buffer_dir))
    
    writer.add_image('sampled_images', img_grid, global_steps)
    writer.add_scalar('Inception_score/mean', mean, global_steps)
    writer.add_scalar('Inception_score/std', std, global_steps)
    writer.add_scalar('FID_score', fid_score, global_steps)

    writer_dict['valid_global_steps'] = global_steps + 1

    return mean, std, fid_score


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten
