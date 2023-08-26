#!/bin/bash
# 13 layers
#layers=('cell1.c0.ops.0.op.1' 'cell1.c1.ops.0.op.1' 'cell1.c2.ops.0.op.1' 'cell1.c3.ops.0.op.1' 'cell2.c0.ops.0.op.1' 'cell2.c2.ops.0.op.1' 'cell2.c3.ops.0.op.1' 'cell2.c4.ops.0.op.1'  'cell3.c0.ops.0.op.1' 'cell3.c1.ops.0.op.1' 'cell3.c2.ops.0.op.1' 'cell3.c3.ops.0.op.1' 'cell3.c4.ops.0.op.1')

# 5 layers
#layers=('cell2.c0.ops.0.op.1' 'cell2.c4.ops.0.op.1'  'cell3.c0.ops.0.op.1' 'cell3.c1.ops.0.op.1' 'cell3.c4.ops.0.op.1')

# 8 layers
layers=('cell1.c0.ops.0.op.1' 'cell1.c1.ops.0.op.1' 'cell1.c2.ops.0.op.1' 'cell1.c3.ops.0.op.1' 'cell2.c2.ops.0.op.1' 'cell2.c3.ops.0.op.1' 'cell3.c2.ops.0.op.1' 'cell3.c3.ops.0.op.1')

dis_layers=('block3.c1' 'block3.c2' 'block4.c1' 'block4.c2' 'l5') # #

gfreeze=true
dfreeze=false

glen=${#layers[@]}
dlen=${#dis_layers[@]}

layers_str=""
for layer in ${layers[@]}; do
    layers_str+=" $layer"
done

dis_layers_str=""
for layer in ${dis_layers[@]}; do
    dis_layers_str+=" $layer"
done

freeze_layers_str=""
if [ "$gfreeze" = true ] ; then
    freeze_layers_str+=$layers_str
fi
if [ "$dfreeze" = true ] ; then
    freeze_layers_str+=" $dis_layers_str"
fi

echo $layers_str
echo $dis_layers_str
echo $freeze_layers_str

bu=4
r=4
R=128
rid=2
lr=0.0002

python MGPU_cpcompress_both_arch_new.py --gpu_ids 0 --num_workers 18 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint arch_train_cifar10_bu4 --genotypes_exp arch_cifar10 --compress-mode "allatonce" --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 200 --exp_name "compress-both-new-gfreeze-dnofreeze-normalizeddecomp-${glen}layers-block3+block4+fc-${R}gr-${r}dr-bu${bu}_lr${lr}_${rid}" --max_epoch_G 1000 --val_freq 5 --byrank --rank ${R} --layers $layers_str --gen_bs  512 --dis_bs 512 --bu $bu --dis_layers $dis_layers_str --dis_rank 256 256 256 256 $r --g_lr $lr --d_lr $lr --freeze_layers $freeze_layers_str #--ft-epochs 100 #--g_lr2 0.000002 --d_lr2 0.000002 #  # --lr_decay   --eval_before_compression
#train arch
#python MGPU_train_arch.py --gpu_ids 0 --num_workers 8 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint arch_train_cifar10_increasebu --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 200 --exp_name "record-info-per-layer-bu${bu}" --max_epoch_G 10 --val_freq 5 --gen_bs  512 --dis_bs 512 --bu $bu #--dis_layers $dis_layers_str --dis_rank $r --eval_before_compression

# train bu 4 architecture
#python MGPU_train_arch.py --gpu_ids 0 --num_workers 8 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False  --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --df_dim 512 --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --num_eval_imgs 50000 --eval_batch_size 200 --exp_name "arch_train_cifar10_bu4_noavgnet" --max_epoch_G 500 --val_freq 5 --gen_bs  512 --dis_bs 512 --bu $bu 

#python MGPU_train_arch.py --gpu_ids 0 --num_workers 8 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False  --genotypes_exp arch_cifar10 --checkpoint arch_train_cifar10_bu4_copy_contnoavg --latent_dim 120 --gf_dim 256 --df_dim 512 --g_lr 0.0002 --d_lr 0.0002 --num_eval_imgs 50000 --eval_batch_size 200  --max_epoch_G 200 --val_freq 5 --gen_bs  512 --dis_bs 512 --bu 4
