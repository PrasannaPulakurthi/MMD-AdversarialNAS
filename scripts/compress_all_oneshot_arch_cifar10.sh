#!/bin/bash


#layers=('cell1.up0.ups.0.c.1' 'cell1.up1.ups.0.c.1' 'cell1.c0.ops.0.op.1' 'cell1.c1.ops.0.op.1' 'cell1.c2.ops.0.op.1' 'cell1.c3.ops.0.op.1' 'cell2.up0.ups.0.c.1' 'cell2.up1.ups.0.c.1' 'cell2.c0.ops.0.op.1' 'cell2.c2.ops.0.op.1' 'cell2.c3.ops.0.op.1' 'cell2.c4.ops.0.op.1' 'cell2.skip_in_ops.0' 'cell3.up0.ups.0.c.1' 'cell3.up1.ups.0.c.1' 'cell3.c0.ops.0.op.1' 'cell3.c1.ops.0.op.1' 'cell3.c2.ops.0.op.1' 'cell3.c3.ops.0.op.1' 'cell3.c4.ops.0.op.1' 'cell3.skip_in_ops.0' 'cell3.skip_in_ops.1')
layers=('cell1.c0.ops.0.op.1' 'cell1.c1.ops.0.op.1' 'cell1.c2.ops.0.op.1' 'cell1.c3.ops.0.op.1' 'cell2.c0.ops.0.op.1' 'cell2.c2.ops.0.op.1' 'cell2.c3.ops.0.op.1' 'cell2.c4.ops.0.op.1'  'cell3.c0.ops.0.op.1' 'cell3.c1.ops.0.op.1' 'cell3.c2.ops.0.op.1' 'cell3.c3.ops.0.op.1' 'cell3.c4.ops.0.op.1')
# only not important layers
#layers=('cell1.c0.ops.0.op.1' 'cell1.c1.ops.0.op.1' 'cell1.c2.ops.0.op.1' 'cell1.c3.ops.0.op.1' 'cell2.c2.ops.0.op.1' 'cell2.c3.ops.0.op.1' 'cell3.c2.ops.0.op.1' 'cell3.c3.ops.0.op.1')
#layers=('cell1.c0.ops.0.op.1')

layers_str=""
for layer in ${layers[@]}; do
    layers_str+=" $layer"
done
#python MGPU_compress_arch.py --gpu_ids 5 --num_workers 8 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint arch_train_cifar10_ncritic1__2023_06_12_15_26_21 --genotypes_exp arch_cifar10 --compress-mode "allatonce" --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 50 --exp_name 'all_layers-ftboth-2' --ft-epochs 1000 --val_freq 10 --byrank --rank 512 --layers $layers_str
#python MGPU_compress_arch.py --gpu_ids 5 --num_workers 8 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10 --compress-mode "allatonce" --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 50 --exp_name 'all_layers-fromscratch ' --ft-epochs 1000 --val_freq 10 --byrank --rank 256 --layers $layers_str
python MGPU_compress_arch.py --gpu_ids 7 --num_workers 8 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint arch_train_cifar10 --genotypes_exp arch_cifar10 --compress-mode "allatonce" --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 200 --exp_name 'CP-13layers-bu128' --ft-epochs 100 --val_freq 5 --byrank --rank 128 --layers $layers_str --gen_bs  256 --dis_bs 256 --bu 128 #--g_lr 0.00001 --d_lr 0.00001
#python MGPU_tucker_arch.py --gpu_ids 7 --num_workers 8 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint arch_train_cifar10_bu4 --genotypes_exp arch_cifar10 --compress-mode "allatonce" --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 200 --exp_name 'compressed_base_all_layers' --ft-epochs 1000 --val_freq 10 --byrank --rank 128 128 128 128 128 128 128 128 256 128 128 128 128 --layers $layers_str --gen_bs  256 --dis_bs 256 --bu 4


# train from scratch
#python MGPU_train_arch.py --gpu_ids 0 --num_workers 8 --gen_bs 40 --dis_bs 80 --dataset cifar10 --bottom_width 4 --img_size 32 --max_epoch_G 200 --n_critic 5 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --df_dim 512 --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --val_freq 5 --num_eval_imgs 50000 --exp_name arch_train_cifar10_
