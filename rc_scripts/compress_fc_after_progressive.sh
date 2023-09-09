#!/bin/bash -l

#SBATCH --job-name=fc    # Job name

# Standard out and Standard Error output files
#SBATCH --output=Results/fc/%x_%j.out   # Instruct Slurm to connect the batch script's standard output directly to the file name specified in the "filename pattern".
#SBATCH --error=Results/fc/%x_%j.err    # Instruct Slurm to connect the batch script's standard error directly to the file name specified in the "filename pattern".

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
#SBATCH --cpus-per-task=18

# Job memory requirements in MB=m (default),GB=g, or TB=t
#SBATCH --mem=32g
#SBATCH --gres=gpu:a100:1

spack env activate tensors-23062101

layers=('l2' 'l3')
r=4
checkpoint_small='CP-compress-cifar10-compress_bigG_nofreeze_256_N_layer_9_13_2023_08_17_03_20_34'
EXPNAME='largeG_R256_N_layer_9_13_r4_l2_l3'
MAXEPOCH=300
BU=4

layers_str=""
for layer in ${layers[@]}; do
    layers_str+=" $layer"
done

python MGPU_cpcompress_arch.py --random_seed 12345 --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint $checkpoint_small --genotypes_exp arch_cifar10 --compress-mode "allatonce" --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name $EXPNAME --max_epoch_G $MAXEPOCH --val_freq 5 --byrank --gen_bs 256 --dis_bs 256 --bu $BU --rank $r --layers $layers_str --eval_before_compression