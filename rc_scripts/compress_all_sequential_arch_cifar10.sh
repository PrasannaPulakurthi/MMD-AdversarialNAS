#!/bin/bash -l

#SBATCH --job-name=L1toL2    # Job name

# Standard out and Standard Error output files
#SBATCH --output=Results/cifar10/%x_%j.out   # Instruct Slurm to connect the batch script's standard output directly to the file name specified in the "filename pattern".
#SBATCH --error=Results/cifar10/%x_%j.err    # Instruct Slurm to connect the batch script's standard error directly to the file name specified in the "filename pattern".

# To send emails, set the adcdress below and remove one of the "#" signs.
#SBATCH --mail-user=mm3424@rit.edu

# notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-type=ALL

# 5 days is the run time MAX, anything over will be KILLED unless you talk with RC
# Time limit days-hrs:min:sec
#SBATCH --time=1-0:0:0

# Put the job in the appropriate partition matchine the account and request FOUR cores

#SBATCH --partition=debug  #currently tier3 is the partition where everyone is put.  To get a listing of partitions where the account can run use the command my-accounts
#SBATCH --ntasks 1  #This option advises the Slurm controller that job steps run within the allocation will launch a maximum of number tasks and to provide for sufficient resources. The default is one task per node.
#SBATCH --cpus-per-task=9

# Job memory requirements in MB=m (default),GB=g, or TB=t
#SBATCH --mem=32g
#SBATCH --gres=gpu:a100:1

spack env activate tensors-23062101

layers=('cell1.c0.ops.0.op.1' 'cell1.c1.ops.0.op.1' 'cell1.c2.ops.0.op.1' 'cell1.c3.ops.0.op.1' 'cell2.c0.ops.0.op.1' 'cell2.c2.ops.0.op.1' 'cell2.c3.ops.0.op.1' 'cell2.c4.ops.0.op.1'  'cell3.c0.ops.0.op.1' 'cell3.c1.ops.0.op.1' 'cell3.c2.ops.0.op.1' 'cell3.c3.ops.0.op.1' 'cell3.c4.ops.0.op.1')

rank=128
# Layer 1
#python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 128 --df_dim 640 --num_eval_imgs 50000 --eval_batch_size 100 --checkpoint arch_train_small_G  --exp_name "compress_noft_${rank}_layer1" --val_freq 5  --gen_bs  256 --dis_bs 256 --beta1 0.0 --beta2 0.9  --byrank --rank $rank --layers cell1.c0.ops.0.op.1 --compress-mode "allatonce" --max_epoch_G 100
# Layer 2
python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 128 --df_dim 640 --num_eval_imgs 50000 --eval_batch_size 100 --checkpoint CP-compress-cifar10-smallG-singlelayercompress-nofreeze-withpartialcopyofoptim-L1-cell1.c0.ops.0.op.1_2023_08_09_09_55_33  --exp_name "compress_sequential_ft_${rank}_layer2" --val_freq 5  --gen_bs  256 --dis_bs 256 --beta1 0.0 --beta2 0.9  --byrank --rank $rank --layers cell1.c1.ops.0.op.1 --compress-mode "allatonce" --max_epoch_G 100
# Layer 3
#python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 128 --df_dim 640 --num_eval_imgs 50000 --eval_batch_size 100 --checkpoint arch_train_small_G  --exp_name "compress_noft_${rank}_layer3" --val_freq 5  --gen_bs  256 --dis_bs 256 --beta1 0.0 --beta2 0.9  --byrank --rank $rank --layers cell1.c2.ops.0.op.1 --compress-mode "allatonce" --max_epoch_G 100
# Layer 4
#python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 128 --df_dim 640 --num_eval_imgs 50000 --eval_batch_size 100 --checkpoint arch_train_small_G  --exp_name "compress_noft_${rank}_layer4" --val_freq 5  --gen_bs  256 --dis_bs 256 --beta1 0.0 --beta2 0.9  --byrank --rank $rank --layers cell1.c3.ops.0.op.1 --compress-mode "allatonce" --max_epoch_G 100
# Layer 5
#python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 128 --df_dim 640 --num_eval_imgs 50000 --eval_batch_size 100 --checkpoint arch_train_small_G  --exp_name "compress_noft_${rank}_layer5" --val_freq 5  --gen_bs  256 --dis_bs 256 --beta1 0.0 --beta2 0.9  --byrank --rank $rank --layers cell2.c0.ops.0.op.1 --compress-mode "allatonce" --max_epoch_G 100
# Layer 6
#python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 128 --df_dim 640 --num_eval_imgs 50000 --eval_batch_size 100 --checkpoint arch_train_small_G  --exp_name "compress_noft_${rank}_layer6" --val_freq 5  --gen_bs  256 --dis_bs 256 --beta1 0.0 --beta2 0.9  --byrank --rank $rank --layers cell2.c2.ops.0.op.1 --compress-mode "allatonce" --max_epoch_G 100
# Layer 7
#python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 128 --df_dim 640 --num_eval_imgs 50000 --eval_batch_size 100 --checkpoint arch_train_small_G  --exp_name "compress_noft_${rank}_layer7" --val_freq 5  --gen_bs  256 --dis_bs 256 --beta1 0.0 --beta2 0.9  --byrank --rank $rank --layers cell2.c3.ops.0.op.1 --compress-mode "allatonce" --max_epoch_G 100
# Layer 8
#python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 128 --df_dim 640 --num_eval_imgs 50000 --eval_batch_size 100 --checkpoint arch_train_small_G  --exp_name "compress_noft_${rank}_layer8" --val_freq 5  --gen_bs  256 --dis_bs 256 --beta1 0.0 --beta2 0.9  --byrank --rank $rank --layers cell2.c4.ops.0.op.1 --compress-mode "allatonce" --max_epoch_G 100
# Layer 9
#python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 128 --df_dim 640 --num_eval_imgs 50000 --eval_batch_size 100 --checkpoint arch_train_small_G  --exp_name "compress_noft_${rank}_layer9" --val_freq 5  --gen_bs  256 --dis_bs 256 --beta1 0.0 --beta2 0.9  --byrank --rank $rank --layers cell3.c0.ops.0.op.1 --compress-mode "allatonce" --max_epoch_G 100
# Layer 10
#python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 128 --df_dim 640 --num_eval_imgs 50000 --eval_batch_size 100 --checkpoint arch_train_small_G  --exp_name "compress_noft_${rank}_layer10" --val_freq 5  --gen_bs  256 --dis_bs 256 --beta1 0.0 --beta2 0.9  --byrank --rank $rank --layers cell3.c1.ops.0.op.1 --compress-mode "allatonce" --max_epoch_G 100
# Layer 11
#python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 128 --df_dim 640 --num_eval_imgs 50000 --eval_batch_size 100 --checkpoint arch_train_small_G  --exp_name "compress_noft_${rank}_layer11" --val_freq 5  --gen_bs  256 --dis_bs 256 --beta1 0.0 --beta2 0.9  --byrank --rank $rank --layers cell3.c2.ops.0.op.1 --compress-mode "allatonce" --max_epoch_G 100
# Layer 12
#python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 128 --df_dim 640 --num_eval_imgs 50000 --eval_batch_size 100 --checkpoint arch_train_small_G  --exp_name "compress_noft_${rank}_layer12" --val_freq 5  --gen_bs  256 --dis_bs 256 --beta1 0.0 --beta2 0.9  --byrank --rank $rank --layers cell3.c3.ops.0.op.1 --compress-mode "allatonce" --max_epoch_G 100
# Layer 13
#python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 128 --df_dim 640 --num_eval_imgs 50000 --eval_batch_size 100 --checkpoint arch_train_small_G  --exp_name "compress_noft_${rank}_layer13" --val_freq 5  --gen_bs  256 --dis_bs 256 --beta1 0.0 --beta2 0.9  --byrank --rank $rank --layers cell3.c4.ops.0.op.1 --compress-mode "allatonce" --max_epoch_G 100

