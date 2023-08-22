#!/bin/bash -l

#SBATCH --job-name=cell    # Job name

# Standard out and Standard Error output files
#SBATCH --output=exps/rc_log/%x_%j.out   # Instruct Slurm to connect the batch script's standard output directly to the file name specified in the "filename pattern".
#SBATCH --error=exps/rc_log/%x_%j.err    # Instruct Slurm to connect the batch script's standard error directly to the file name specified in the "filename pattern".

# To send emails, set the adcdress below and remove one of the "#" signs.
#SBATCH --mail-user=mm3424@rit.edu

# notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-type=ALL

# 5 days is the run time MAX, anything over will be KILLED unless you talk with RC
# Time limit days-hrs:min:sec
#SBATCH --time=5-0:0:0

# Put the job in the appropriate partition matchine the account and request FOUR cores

#SBATCH --partition=tier3  #currently tier3 is the partition where everyone is put.  To get a listing of partitions where the account can run use the command my-accounts
#SBATCH --ntasks 1  #This option advises the Slurm controller that job steps run within the allocation will launch a maximum of number tasks and to provide for sufficient resources. The default is one task per node.
#SBATCH --cpus-per-task=18

# Job memory requirements in MB=m (default),GB=g, or TB=t
#SBATCH --mem=32g
#SBATCH --gres=gpu:a100:1

spack env activate tensors-23062101


layers=('cell1.c0.ops.0.op.1' 'cell1.c1.ops.0.op.1' 'cell1.c2.ops.0.op.1' 'cell1.c3.ops.0.op.1' 'cell2.c0.ops.0.op.1' 'cell2.c2.ops.0.op.1' 'cell2.c3.ops.0.op.1' 'cell2.c4.ops.0.op.1'  'cell3.c0.ops.0.op.1' 'cell3.c1.ops.0.op.1' 'cell3.c2.ops.0.op.1' 'cell3.c3.ops.0.op.1' 'cell3.c4.ops.0.op.1')

rank=256
checkpoint="arch_train_cifar10_smallG"
### Small Network

# Entire Network
#python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 128 --df_dim 640 --num_eval_imgs 50000 --eval_batch_size 100 --checkpoint $checkpoint  --exp_name "compress_smallG_nofreeze_${rank}_Entire" --val_freq 5  --gen_bs  256 --dis_bs 256 --beta1 0.0 --beta2 0.9  --byrank --rank $rank --layers cell1.c0.ops.0.op.1 cell1.c1.ops.0.op.1 cell1.c2.ops.0.op.1 cell1.c3.ops.0.op.1 cell2.c0.ops.0.op.1 cell2.c2.ops.0.op.1 cell2.c3.ops.0.op.1 cell2.c4.ops.0.op.1 cell3.c0.ops.0.op.1 cell3.c1.ops.0.op.1 cell3.c2.ops.0.op.1 cell3.c3.ops.0.op.1 cell3.c4.ops.0.op.1 --compress-mode "allatonce" --max_epoch_G  500 --eval_before_compression

# N_Layer 9_13
#python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 128 --df_dim 640 --num_eval_imgs 50000 --eval_batch_size 100 --checkpoint $checkpoint  --exp_name "compress_smallG_nofreeze_${rank}_N_layer_9_13" --val_freq 5  --gen_bs  256 --dis_bs 256 --beta1 0.0 --beta2 0.9  --byrank --rank $rank --layers cell1.c0.ops.0.op.1 cell1.c1.ops.0.op.1 cell1.c2.ops.0.op.1 cell1.c3.ops.0.op.1 cell2.c0.ops.0.op.1 cell2.c2.ops.0.op.1 cell2.c3.ops.0.op.1 cell2.c4.ops.0.op.1 cell3.c1.ops.0.op.1 cell3.c2.ops.0.op.1 cell3.c3.ops.0.op.1 --compress-mode "allatonce" --max_epoch_G 400 --eval_before_compression

# N_Layer 5_9_13
#python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 128 --df_dim 640 --num_eval_imgs 50000 --eval_batch_size 100 --checkpoint $checkpoint  --exp_name "compress_smallG_nofreeze_${rank}_N_layer_5_9_13" --val_freq 5  --gen_bs  256 --dis_bs 256 --beta1 0.0 --beta2 0.9  --byrank --rank $rank --layers cell1.c0.ops.0.op.1 cell1.c1.ops.0.op.1 cell1.c2.ops.0.op.1 cell1.c3.ops.0.op.1 cell2.c2.ops.0.op.1 cell2.c3.ops.0.op.1 cell2.c4.ops.0.op.1 cell3.c1.ops.0.op.1 cell3.c2.ops.0.op.1 cell3.c3.ops.0.op.1 --compress-mode "allatonce" --max_epoch_G 300 --eval_before_compression

# Cell 1
#python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 128 --df_dim 640 --num_eval_imgs 50000 --eval_batch_size 100 --checkpoint $checkpoint  --exp_name "compress_smallG_nofreeze_${rank}_cell1" --val_freq 5  --gen_bs  256 --dis_bs 256 --beta1 0.0 --beta2 0.9  --byrank --rank $rank --layers cell1.c0.ops.0.op.1 cell1.c1.ops.0.op.1 cell1.c2.ops.0.op.1 cell1.c3.ops.0.op.1 --compress-mode "allatonce" --max_epoch_G 300 --eval_before_compression
# Cell 2
#python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 128 --df_dim 640 --num_eval_imgs 50000 --eval_batch_size 100 --checkpoint $checkpoint  --exp_name "compress_smallG_nofreeze_${rank}_cell2" --val_freq 5  --gen_bs  256 --dis_bs 256 --beta1 0.0 --beta2 0.9  --byrank --rank $rank --layers cell2.c0.ops.0.op.1 cell2.c2.ops.0.op.1 cell2.c3.ops.0.op.1 cell2.c4.ops.0.op.1 --compress-mode "allatonce" --max_epoch_G 300  --eval_before_compression
# Cell 3
#python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 128 --df_dim 640 --num_eval_imgs 50000 --eval_batch_size 100 --checkpoint $checkpoint  --exp_name "compress_smallG_nofreeze_${rank}_cell3" --val_freq 5  --gen_bs  256 --dis_bs 256 --beta1 0.0 --beta2 0.9  --byrank --rank $rank --layers cell3.c0.ops.0.op.1 cell3.c1.ops.0.op.1 cell3.c2.ops.0.op.1 cell3.c3.ops.0.op.1 cell3.c4.ops.0.op.1 --compress-mode "allatonce" --max_epoch_G 300  --eval_before_compression


# Cell 1 & 2
#python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 128 --df_dim 640 --num_eval_imgs 50000 --eval_batch_size 100 --checkpoint $checkpoint  --exp_name "compress_smallG_nofreeze_${rank}_cell1_2" --val_freq 5  --gen_bs  256 --dis_bs 256 --beta1 0.0 --beta2 0.9  --byrank --rank $rank --layers cell1.c0.ops.0.op.1 cell1.c1.ops.0.op.1 cell1.c2.ops.0.op.1 cell1.c3.ops.0.op.1 cell2.c0.ops.0.op.1 cell2.c2.ops.0.op.1 cell2.c3.ops.0.op.1 cell2.c4.ops.0.op.1 --compress-mode "allatonce" --max_epoch_G 400

##### Big Network

checkpoint_big="arch_train_cifar10_increasebu"
# Entire Network
#python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 100 --checkpoint $checkpoint_big  --exp_name "compress_bigG_nofreeze_${rank}_Entire" --val_freq 5  --gen_bs  256 --dis_bs 256 --beta1 0.0 --beta2 0.9  --byrank --rank $rank --layers cell1.c0.ops.0.op.1 cell1.c1.ops.0.op.1 cell1.c2.ops.0.op.1 cell1.c3.ops.0.op.1 cell2.c0.ops.0.op.1 cell2.c2.ops.0.op.1 cell2.c3.ops.0.op.1 cell2.c4.ops.0.op.1 cell3.c0.ops.0.op.1 cell3.c1.ops.0.op.1 cell3.c2.ops.0.op.1 cell3.c3.ops.0.op.1 cell3.c4.ops.0.op.1 --compress-mode "allatonce" --max_epoch_G  500 --eval_before_compression

# N_Layer 9_13
#python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 100 --checkpoint $checkpoint_big  --exp_name "compress_bigG_nofreeze_${rank}_N_layer_9_13" --val_freq 5  --gen_bs  256 --dis_bs 256 --beta1 0.0 --beta2 0.9  --byrank --rank $rank --layers cell1.c0.ops.0.op.1 cell1.c1.ops.0.op.1 cell1.c2.ops.0.op.1 cell1.c3.ops.0.op.1 cell2.c0.ops.0.op.1 cell2.c2.ops.0.op.1 cell2.c3.ops.0.op.1 cell2.c4.ops.0.op.1 cell3.c1.ops.0.op.1 cell3.c2.ops.0.op.1 cell3.c3.ops.0.op.1 --compress-mode "allatonce" --max_epoch_G 400 --eval_before_compression

# N_Layer 5_9_13
#python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 100 --checkpoint $checkpoint_big  --exp_name "compress_bigG_nofreeze_${rank}_N_layer_5_9_13" --val_freq 5  --gen_bs  256 --dis_bs 256 --beta1 0.0 --beta2 0.9  --byrank --rank $rank --layers cell1.c0.ops.0.op.1 cell1.c1.ops.0.op.1 cell1.c2.ops.0.op.1 cell1.c3.ops.0.op.1 cell2.c2.ops.0.op.1 cell2.c3.ops.0.op.1 cell2.c4.ops.0.op.1 cell3.c1.ops.0.op.1 cell3.c2.ops.0.op.1 cell3.c3.ops.0.op.1 --compress-mode "allatonce" --max_epoch_G 300 --eval_before_compression


# Cell 1
#python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 100 --checkpoint $checkpoint_big  --exp_name "compress_bigG_nofreeze_${rank}_cell1" --val_freq 5  --gen_bs  256 --dis_bs 256 --beta1 0.0 --beta2 0.9  --byrank --rank $rank --layers cell1.c0.ops.0.op.1 cell1.c1.ops.0.op.1 cell1.c2.ops.0.op.1 cell1.c3.ops.0.op.1 --compress-mode "allatonce" --max_epoch_G 300  --eval_before_compression
# Cell 2
#python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 100 --checkpoint $checkpoint_big  --exp_name "compress_bigG_nofreeze_${rank}_cell2" --val_freq 5  --gen_bs  256 --dis_bs 256 --beta1 0.0 --beta2 0.9  --byrank --rank $rank --layers cell2.c0.ops.0.op.1 cell2.c2.ops.0.op.1 cell2.c3.ops.0.op.1 cell2.c4.ops.0.op.1 --compress-mode "allatonce" --max_epoch_G 300  --eval_before_compression
# Cell 3
#python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 100 --checkpoint $checkpoint_big  --exp_name "compress_bigG_nofreeze_${rank}_cell3" --val_freq 5  --gen_bs  256 --dis_bs 256 --beta1 0.0 --beta2 0.9  --byrank --rank $rank --layers cell3.c0.ops.0.op.1 cell3.c1.ops.0.op.1 cell3.c2.ops.0.op.1 cell3.c3.ops.0.op.1 cell3.c4.ops.0.op.1 --compress-mode "allatonce" --max_epoch_G 300 --eval_before_compression


# Cell 1 & 2
python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 100 --checkpoint $checkpoint_big  --exp_name "compress_bigG_nofreeze_${rank}_cell1_2" --val_freq 5  --gen_bs  256 --dis_bs 256 --beta1 0.0 --beta2 0.9  --byrank --rank $rank --layers cell1.c0.ops.0.op.1 cell1.c1.ops.0.op.1 cell1.c2.ops.0.op.1 cell1.c3.ops.0.op.1 cell2.c0.ops.0.op.1 cell2.c2.ops.0.op.1 cell2.c3.ops.0.op.1 cell2.c4.ops.0.op.1 --compress-mode "allatonce" --max_epoch_G 400
