


lr=0.0001
bu=4
ncritic=10

python MGPU_train_arch.py --gpu_ids 5 --num_workers 8 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint Tucker_compress-allatonce-cifar10-byrank128-compressed_base_all_layers --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 200 --exp_name "train_compressed_tucker_ncritic_${ncritic}_lr${lr}-bu${bu}" --val_freq 5  --gen_bs  256 --dis_bs 256 --bu ${bu} --g_lr ${lr} --d_lr ${lr} --beta1 0.0 --beta2 0.9 --n_critic ${ncritic}