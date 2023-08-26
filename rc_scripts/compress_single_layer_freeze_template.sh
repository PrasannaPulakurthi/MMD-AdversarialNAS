#!/bin/bash -l

#SBATCH --job-name=compL13    # Job name

# Standard out and Standard Error output files
#SBATCH --output=exps/rc_log/%x_%j.out   # Instruct Slurm to connect the batch script's standard output directly to the file name specified in the "filename pattern".
#SBATCH --error=exps/rc_log/%x_%j.err    # Instruct Slurm to connect the batch script's standard error directly to the file name specified in the "filename pattern".

# To send emails, set the adcdress below and remove one of the "#" signs.
#SBATCH --mail-user=mm3424@rit.edu

# notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-type=ALL

# 5 days is the run time MAX, anything over will be KILLED unless you talk with RC
# Time limit days-hrs:min:sec
#SBATCH --time=0-1:0:0

# Put the job in the appropriate partition matchine the account and request FOUR cores

#SBATCH --partition=tier3  #currently tier3 is the partition where everyone is put.  To get a listing of partitions where the account can run use the command my-accounts
#SBATCH --ntasks 1  #This option advises the Slurm controller that job steps run within the allocation will launch a maximum of number tasks and to provide for sufficient resources. The default is one task per node.
#SBATCH --cpus-per-task=18

# Job memory requirements in MB=m (default),GB=g, or TB=t
#SBATCH --mem=32g
#SBATCH --gres=gpu:a100:1

spack env activate tensors-23062101

layers_str=""
for layer in ${layers[@]}; do
    layers_str+=" $layer"
done
echo $FREEZE_TYPE

if [ "$FREEZE_TYPE" = "reverse" ]; then
    echo 'reverse freeze'
    python MGPU_cpcompress_arch.py --random_seed 12345 --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint $CHECKPOINT --genotypes_exp arch_cifar10 --compress-mode "allatonce" --latent_dim 120 --gf_dim 128 --df_dim 640 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name $EXPNAME --max_epoch_G $MAXEPOCH --val_freq 5 --byrank --rank $R --layers $LAYER --gen_bs 256 --dis_bs 256 --bu $BU --freeze_layers $FREEZE_LAYERS --reverse_g_freeze --eval_before_compression
elif [ "$FREEZE_TYPE" = "before" ]; then
    echo 'freeze before'
    python MGPU_cpcompress_arch.py --random_seed 12345 --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint $CHECKPOINT --genotypes_exp arch_cifar10 --compress-mode "allatonce" --latent_dim 120 --gf_dim 128 --df_dim 640 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name $EXPNAME --max_epoch_G $MAXEPOCH --val_freq 5 --byrank --rank $R --layers $LAYER --gen_bs 256 --dis_bs 256 --bu $BU  --freeze_before_compressed --eval_before_compression
else
    echo 'freeze compressed'
    python MGPU_cpcompress_arch.py --random_seed 12345 --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint $CHECKPOINT --genotypes_exp arch_cifar10 --compress-mode "allatonce" --latent_dim 120 --gf_dim 128 --df_dim 640 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name $EXPNAME --max_epoch_G $MAXEPOCH --val_freq 5 --byrank --rank $R --layers $LAYER --gen_bs 256 --dis_bs 256 --bu $BU --freeze_layers $FREEZE_LAYERS --eval_before_compression
fi

#if [ "$REVERSE_FREEZE" = true ] ; then
#    python MGPU_cpcompress_arch.py --random_seed 12345 --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint $CHECKPOINT --genotypes_exp arch_cifar10 --compress-mode "allatonce" --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 200 --exp_name $EXPNAME --max_epoch_G $MAXEPOCH --val_freq 5 --byrank --rank $R --layers $LAYER --gen_bs 256 --dis_bs 256 --bu $BU --freeze_layers $FREEZE_LAYERS --reverse_g_freeze
#fi 


#else
#    python MGPU_cpcompress_arch.py --random_seed 12345 --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint $CHECKPOINT --genotypes_exp arch_cifar10 --compress-mode "allatonce" --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 200 --exp_name $EXPNAME --max_epoch_G $MAXEPOCH --val_freq 5 --byrank --rank $R --layers $LAYER --gen_bs 256 --dis_bs 256 --bu $BU --freeze_layers $FREEZE_LAYERS
#fi


