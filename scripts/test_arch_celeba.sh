
# Large Network
python MGPU_test_arch.py --gpu_ids 0 --num_workers 8 --dataset celeba --bottom_width 8 --img_size 64 --arch arch_cifar10 --draw_arch False --checkpoint arch_train_celeba_large --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name arch_test_celeba_large

# Small Network
python MGPU_test_arch.py --gpu_ids 0 --num_workers 8 --dataset celeba --bottom_width 8 --img_size 64 --arch arch_cifar10 --draw_arch False --checkpoint arch_train_celeba_small --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 128 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name arch_test_celeba_small