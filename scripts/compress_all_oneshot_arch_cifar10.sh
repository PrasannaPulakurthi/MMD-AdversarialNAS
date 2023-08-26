#!/bin/bash
# all dis layers
#dis_layers=('block1.c2' 'block2.c1' 'block2.c2' 'block3.c1' 'block3.c2' 'block4.c1' 'block4.c2')
dis_layers=('l5')


#layers=('cell1.up0.ups.0.c.1' 'cell1.up1.ups.0.c.1' 'cell1.c0.ops.0.op.1' 'cell1.c1.ops.0.op.1' 'cell1.c2.ops.0.op.1' 'cell1.c3.ops.0.op.1' 'cell2.up0.ups.0.c.1' 'cell2.up1.ups.0.c.1' 'cell2.c0.ops.0.op.1' 'cell2.c2.ops.0.op.1' 'cell2.c3.ops.0.op.1' 'cell2.c4.ops.0.op.1' 'cell2.skip_in_ops.0' 'cell3.up0.ups.0.c.1' 'cell3.up1.ups.0.c.1' 'cell3.c0.ops.0.op.1' 'cell3.c1.ops.0.op.1' 'cell3.c2.ops.0.op.1' 'cell3.c3.ops.0.op.1' 'cell3.c4.ops.0.op.1' 'cell3.skip_in_ops.0' 'cell3.skip_in_ops.1')
#13 layers:
#layers=('cell1.c0.ops.0.op.1' 'cell1.c1.ops.0.op.1' 'cell1.c2.ops.0.op.1' 'cell1.c3.ops.0.op.1' 'cell2.c0.ops.0.op.1' 'cell2.c2.ops.0.op.1' 'cell2.c3.ops.0.op.1' 'cell2.c4.ops.0.op.1'  'cell3.c0.ops.0.op.1' 'cell3.c1.ops.0.op.1' 'cell3.c2.ops.0.op.1' 'cell3.c3.ops.0.op.1' 'cell3.c4.ops.0.op.1')
# only not important layers (8 layers)
layers=('cell1.c0.ops.0.op.1' 'cell1.c1.ops.0.op.1' 'cell1.c2.ops.0.op.1' 'cell1.c3.ops.0.op.1' 'cell2.c2.ops.0.op.1' 'cell2.c3.ops.0.op.1' 'cell3.c2.ops.0.op.1' 'cell3.c3.ops.0.op.1')
#layers2=('cell1.c0.ops.0.op.1' 'cell1.c1.ops.0.op.1' 'cell1.c2.ops.0.op.1' 'cell1.c3.ops.0.op.1' 'cell2.c2.ops.0.op.1' 'cell2.c3.ops.0.op.1' 'cell3.c2.ops.0.op.1' 'cell3.c3.ops.0.op.1')

# only important layers (5 layers)
#layers=('cell2.c0.ops.0.op.1' 'cell2.c4.ops.0.op.1'  'cell3.c0.ops.0.op.1' 'cell3.c1.ops.0.op.1' 'cell3.c4.ops.0.op.1')

layers_str=""
for layer in ${layers[@]}; do
    layers_str+=" $layer"
done

#freeze_layers_str=""
#for layer in ${layers2[@]}; do
#    freeze_layers_str+=" $layer"
#done

dis_layers_str=""
for layer in ${dis_layers[@]}; do
    dis_layers_str+=" $layer"
done

#python MGPU_compress_arch.py --gpu_ids 5 --num_workers 8 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint arch_train_cifar10_ncritic1__2023_06_12_15_26_21 --genotypes_exp arch_cifar10 --compress-mode "allatonce" --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 50 --exp_name 'all_layers-ftboth-2' --ft-epochs 1000 --val_freq 10 --byrank --rank 512 --layers $layers_str
#python MGPU_compress_arch.py --gpu_ids 5 --num_workers 8 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10 --compress-mode "allatonce" --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 50 --exp_name 'all_layers-fromscratch ' --ft-epochs 1000 --val_freq 10 --byrank --rank 256 --layers $layers_str
#python MGPU_compress_arch.py --gpu_ids 2 --num_workers 8 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint arch_train_cifar10 --genotypes_exp arch_cifar10 --compress-mode "allatonce" --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 200 --exp_name 'CP-5layers-bu128' --ft-epochs 1000 --val_freq 10 --byrank --rank 128 --layers $layers_str --gen_bs  256 --dis_bs 512 --bu 128 #--g_lr 0.00001 --d_lr 0.00001
#python MGPU_compress_continue_arch.py --gpu_ids 7 --num_workers 8 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint compress-allatonce-cifar10-byrank128-CP-5layers-bu128_2023_06_25_22_00_10 --genotypes_exp arch_cifar10 --compress-mode "allatonce" --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 200 --exp_name 'CP-5layers-bu128-continue' --ft-epochs 1000 --val_freq 10 --byrank --rank 128 --layers $layers_str --gen_bs  256 --dis_bs 512 --bu 128 #--g_lr 0.00001 --d_lr 0.00001
#python MGPU_tucker_arch.py --gpu_ids 7 --num_workers 8 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint arch_train_cifar10_bu4 --genotypes_exp arch_cifar10 --compress-mode "allatonce" --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 200 --exp_name 'compressed_base_all_layers' --ft-epochs 1000 --val_freq 10 --byrank --rank 128 128 128 128 128 128 128 128 256 128 128 128 128 --layers $layers_str --gen_bs  256 --dis_bs 256 --bu 4

#python MGPU_compress_arch.py --gpu_ids 7 --num_workers 8 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint arch_train_cifar10_bu4 --genotypes_exp arch_cifar10 --compress-mode "allatonce" --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 200 --exp_name 'CP-5layers-bu4' --ft-epochs 1000 --val_freq 10 --byrank --rank 128 --layers $layers_str --gen_bs  512 --dis_bs 512 --bu 4
#python MGPU_cpcompress_arch.py --gpu_ids 7 --num_workers 8 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint CP-compress-cifar10-CP-8layers-bu4-newcode_2023_06_29_04_00_56 --genotypes_exp arch_cifar10 --compress-mode "allatonce" --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 200 --exp_name 'CP-8layers-bu4-newcode-continue(lr1e-6)' --ft-epochs 100 --val_freq 5 --byrank  --gen_bs  512 --dis_bs 512 --bu 4 --g_lr 0.000001 --d_lr 0.000001 # --rank 128 --layers $layers_str
#python MGPU_cpcompress_arch.py --gpu_ids 3 --num_workers 8 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint arch_train_cifar10_bu4 --genotypes_exp arch_cifar10 --compress-mode "allatonce" --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 200 --exp_name 'CP-8layers-partial-freeze' --ft-epochs 1000 --val_freq 10 --byrank --rank 128 --layers $layers_str --gen_bs  512 --dis_bs 512 --bu 4 --freeze_layers $freeze_layers_str --eval_before_compression

# train from scratch
#python MGPU_train_arch.py --gpu_ids 0 --num_workers 8 --gen_bs 40 --dis_bs 80 --dataset cifar10 --bottom_width 4 --img_size 32 --max_epoch_G 200 --n_critic 5 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --df_dim 512 --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --val_freq 5 --num_eval_imgs 50000 --exp_name arch_train_cifar10_

#
#python MGPU_test_arch.py --gpu_ids 7 --num_workers 8 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint arch_train_cifar10_bu4 --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 200 --exp_name 'base-test-bu4'  --val_freq 10 --gen_bs  512 --dis_bs 512 --bu 4 

# compress both
#python MGPU_cpcompress_both_arch.py --gpu_ids 4 --num_workers 8 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint arch_train_cifar10_bu4 --genotypes_exp arch_cifar10 --compress-mode "allatonce" --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 200 --exp_name 'compress-both-8layers-fc-gr128-dr-10-bu4' --ft-epochs 10000 --val_freq 5 --byrank  --gen_bs  512 --dis_bs 512 --bu 4  --max_epoch_G 10000 --rank 128 --layers $layers_str --dis_layers $dis_layers_str --dis_rank  10 #--g_lr 0.000001 --d_lr 0.000001 # --rank 128 --layers $layers_str
python MGPU_cpcompress_both_arch.py --gpu_ids 7 --num_workers 8 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint CP-compress-cifar10-compress-both-8layers-fc-gr128-dr-10-bu4 --genotypes_exp arch_cifar10 --compress-mode "allatonce" --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 200 --exp_name 'compress-both-8layers-fc-gr128-dr-10-continue-bu-increase-to-bu8' --ft-epochs 1000 --val_freq 5 --byrank  --gen_bs  512 --dis_bs 512 --bu 8  --max_epoch_G 1000 #--rank 128 --layers $layers_str --dis_layers $dis_layers_str --dis_rank  6 #--g_lr 0.000001 --d_lr 0.000001 # --rank 128 --layers $layers_str
#python MGPU_cpcompress_both_arch.py --gpu_ids 7 --num_workers 8 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint CP-compress-cifar10-CP-8layers-3layers-bu4-bothnets_2023_07_07_20_23_42 --genotypes_exp arch_cifar10 --compress-mode "allatonce" --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 200 --exp_name '' --ft-epochs 10000 --val_freq 5 --byrank  --gen_bs  512 --dis_bs 512 --bu 4 --resume --max_epoch_G 10000 --current # --rank 128 --layers $layers_str --dis_layers $dis_layers_str --dis_rank  128 #--g_lr 0.000001 --d_lr 0.000001 # --rank 128 --layers $layers_str