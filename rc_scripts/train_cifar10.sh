#!/bin/bash -l

#SBATCH --job-name=cifar10_L    # Job name

# Standard out and Standard Error output files
#SBATCH --output=Results/cifar10/%x_%j.out   # Instruct Slurm to connect the batch script's standard output directly to the file name specified in the "filename pattern".
#SBATCH --error=Results/cifar10/%x_%j.err    # Instruct Slurm to connect the batch script's standard error directly to the file name specified in the "filename pattern".

# To send emails, set the adcdress below and remove one of the "#" signs.
#SBATCH --mail-user=pp4405@rit.edu

# notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-type=ALL

# 5 days is the run time MAX, anything over will be KILLED unless you talk with RC
# Time limit days-hrs:min:sec
#SBATCH --time=2-0:0:0

# Put the job in the appropriate partition matchine the account and request FOUR cores
#SBATCH --account=vividgan   #this is the name created for your project when you filled out the questionnaire
#SBATCH --partition=tier3  #currently tier3 is the partition where everyone is put.  To get a listing of partitions where the account can run use the command my-accounts
#SBATCH --ntasks 1  #This option advises the Slurm controller that job steps run within the allocation will launch a maximum of number tasks and to provide for sufficient resources. The default is one task per node.
#SBATCH --cpus-per-task=18

# Job memory requirements in MB=m (default),GB=g, or TB=t
#SBATCH --mem=32g
#SBATCH --gres=gpu:a100:1

spack env activate tensors-23062101

# Train Large Network
python MGPU_train_arch.py --gpu_ids 0 --num_workers 18 --gen_bs 128 --dis_bs 128 --dataset cifar10 --bottom_width 4 --img_size 32 --max_epoch_G 500 --n_critic 1 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --df_dim 512 --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --val_freq 5 --num_eval_imgs 50000 --exp_name arch_train_cifar10_large

# Train Small Network
# python MGPU_train_arch.py --gpu_ids 0 --num_workers 18 --gen_bs 128 --dis_bs 128 --dataset cifar10 --bottom_width 4 --img_size 32 --max_epoch_G 500 --n_critic 1 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 128 --df_dim 512 --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --val_freq 5 --num_eval_imgs 50000 --exp_name arch_train_cifar10_small