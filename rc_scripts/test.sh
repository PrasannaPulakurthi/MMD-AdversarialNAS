#!/bin/bash -l

#SBATCH --job-name=comp    # Job name

# Standard out and Standard Error output files
#SBATCH --output=Results/cifar10/%x_%j.out   # Instruct Slurm to connect the batch script's standard output directly to the file name specified in the "filename pattern".
#SBATCH --error=Results/cifar10/%x_%j.err    # Instruct Slurm to connect the batch script's standard error directly to the file name specified in the "filename pattern".

# To send emails, set the adcdress below and remove one of the "#" signs.
#SBATCH --mail-user=mm3424@rit.edu

# notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-type=ALL

# 5 days is the run time MAX, anything over will be KILLED unless you talk with RC
# Time limit days-hrs:min:sec
#SBATCH --time=0-1:0:0

# Put the job in the appropriate partition matchine the account and request FOUR cores

#SBATCH --partition=debug  #currently tier3 is the partition where everyone is put.  To get a listing of partitions where the account can run use the command my-accounts
#SBATCH --ntasks 1  #This option advises the Slurm controller that job steps run within the allocation will launch a maximum of number tasks and to provide for sufficient resources. The default is one task per node.
#SBATCH --cpus-per-task=9

# Job memory requirements in MB=m (default),GB=g, or TB=t
#SBATCH --mem=32g
#SBATCH --gres=gpu:a100:1


spack env activate tensors-23062101

# 13 layers



python MGPU_test_arch.py --gpu_ids 0 --num_workers 8 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint arch_train_cifar10_bu4 --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name arch_test_cifar10_bu4_no_avgnet --bu 4 