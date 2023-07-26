#!/bin/bash -l

#SBATCH --job-name=cmp8+1fc-nffc    # Job name

# Standard out and Standard Error output files
#SBATCH --output=Results/cifar10/%x_%j.out   # Instruct Slurm to connect the batch script's standard output directly to the file name specified in the "filename pattern".
#SBATCH --error=Results/cifar10/%x_%j.err    # Instruct Slurm to connect the batch script's standard error directly to the file name specified in the "filename pattern".

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
#SBATCH --cpus-per-task=9

# Job memory requirements in MB=m (default),GB=g, or TB=t
#SBATCH --mem=32g
#SBATCH --gres=gpu:a100:1


spack env activate tensors-23062101

# 13 layers
#layers=('cell1.c0.ops.0.op.1' 'cell1.c1.ops.0.op.1' 'cell1.c2.ops.0.op.1' 'cell1.c3.ops.0.op.1' 'cell2.c0.ops.0.op.1' 'cell2.c2.ops.0.op.1' 'cell2.c3.ops.0.op.1' 'cell2.c4.ops.0.op.1'  'cell3.c0.ops.0.op.1' 'cell3.c1.ops.0.op.1' 'cell3.c2.ops.0.op.1' 'cell3.c3.ops.0.op.1' 'cell3.c4.ops.0.op.1')

# 5 layers
#layers=('cell2.c0.ops.0.op.1' 'cell2.c4.ops.0.op.1'  'cell3.c0.ops.0.op.1' 'cell3.c1.ops.0.op.1' 'cell3.c4.ops.0.op.1')

# 8 layers
layers=('cell1.c0.ops.0.op.1' 'cell1.c1.ops.0.op.1' 'cell1.c2.ops.0.op.1' 'cell1.c3.ops.0.op.1' 'cell2.c2.ops.0.op.1' 'cell2.c3.ops.0.op.1' 'cell3.c2.ops.0.op.1' 'cell3.c3.ops.0.op.1')

dis_layers=('block3.c1' 'block3.c2' 'block4.c1' 'block4.c2' 'l5') # #

layers_str=""
for layer in ${layers[@]}; do
    layers_str+=" $layer"
done

dis_layers_str=""
for layer in ${dis_layers[@]}; do
    dis_layers_str+=" $layer"
done
freeze_layers_str=""
freeze_layers_str+=$layers_str
#freeze_layers_str+=" $dis_layers_str"

echo $layers_str
echo $dis_layers_str
echo $freeze_layers_str

bu=4
r=4
R=128
rid=1
lr=0.0002

python MGPU_cpcompress_both_arch_new.py --gpu_ids 0 --num_workers 8 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint arch_train_cifar10_bu4 --genotypes_exp arch_cifar10 --compress-mode "allatonce" --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 200 --exp_name "compress-both-new-gfreeze-dnofreeze-normalizeddecomp-13layers-block3+block4+fc-${R}gr-${r}dr-bu${bu}_lr${lr}_${rid}" --max_epoch_G 1000 --val_freq 5 --byrank --rank ${R} --layers $layers_str --gen_bs  512 --dis_bs 512 --bu $bu --dis_layers $dis_layers_str --dis_rank 256 256 256 256 $r --g_lr $lr --d_lr $lr --freeze_layers $freeze_layers_str #--ft-epochs 100 #--g_lr2 0.000002 --d_lr2 0.000002 #  # --lr_decay   --eval_before_compression
#train arch
#python MGPU_train_arch.py --gpu_ids 0 --num_workers 8 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint arch_train_cifar10_increasebu --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 200 --exp_name "record-info-per-layer-bu${bu}" --max_epoch_G 10 --val_freq 5 --gen_bs  512 --dis_bs 512 --bu $bu #--dis_layers $dis_layers_str --dis_rank $r --eval_before_compression

# train bu 4 architecture
#python MGPU_train_arch.py --gpu_ids 0 --num_workers 8 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False  --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --df_dim 512 --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --num_eval_imgs 50000 --eval_batch_size 200 --exp_name "arch_train_cifar10_bu4_noavgnet" --max_epoch_G 500 --val_freq 5 --gen_bs  512 --dis_bs 512 --bu $bu 

#python MGPU_train_arch.py --gpu_ids 0 --num_workers 8 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False  --genotypes_exp arch_cifar10 --checkpoint arch_train_cifar10_bu4_copy_contnoavg --latent_dim 120 --gf_dim 256 --df_dim 512 --g_lr 0.0002 --d_lr 0.0002 --num_eval_imgs 50000 --eval_batch_size 200  --max_epoch_G 200 --val_freq 5 --gen_bs  512 --dis_bs 512 --bu 4
