# <p align="center">ENHANCING GAN PERFORMANCE THROUGH NEURAL ARCHITECTURE SEARCH AND TENSOR DECOMPOSITION (MMD-AdversarialNAS)</p>

<p align="center">
  <a href="https://ieeexplore.ieee.org/document/10446488"><img src="https://img.shields.io/badge/IEEE%20ICASSP%202024-Paper-blue" alt="IEEE ICASSP 2024"></a>
  <a href="https://paperswithcode.com/paper/enhancing-gan-performance-through-neural"><img src="https://img.shields.io/badge/Papers%20with%20Code-MMD--AdversarialNAS-blue" alt="Papers with Code"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

This repository contains code for our **ICASSP 2024** paper "**Enhancing GAN Performance Through Neural Architecture Search and Tensor Decomposition.**"
by [Prasanna Reddy Pulakurthi](https://www.prasannapulakurthi.com/), [Mahsa Mozaffari](https://mahsamozaffari.com/), [Sohail A. Dianat](https://www.rit.edu/directory/sadeee-sohail-dianat), [Majid Rabbani](https://www.rit.edu/directory/mxreee-majid-rabbani), [Jamison Heard](https://www.rit.edu/directory/jrheee-jamison-heard), and [Raghuveer Rao](https://ieeexplore.ieee.org/author/37281258600). [[PDF]](https://prasannapulakurthi.github.io/papers/PDFs/2024_ICASSP_GANs-Tensor-Decomposition.pdf) [[PPT]](https://sigport.org/documents/enhancing-gan-performance-through-neural-architecture-search-and-tensor-decomposition) [[Paper]](https://ieeexplore.ieee.org/document/10446488)

> **Update (November 2024):**  
> An extended version of this work, titled **"Enhancing GANs with MMD Neural Architecture Search, PMish Activation Function, and Adaptive Rank Decomposition,"** has been accepted for publication in **IEEE Access 2024**. 🔗 [Paper](https://ieeexplore.ieee.org/document/10732016) | 🌐 [Project Website](https://prasannapulakurthi.github.io/mmdpmishnas/)  
> The latest version of the code corresponding to the extended paper is available [here](https://github.com/PrasannaPulakurthi/MMD-PMish-NAS).

## Qualitative Results
![All Visual Results](assets/All_Grid1.png)

## Quantitative Results
![Quantitative Results](assets/Quantitative_Results.png)

## Repeatability
![Reproducibility Results](assets/Reproducibility.png)

## Getting Started
### Installation
1. Clone this repository.

    ~~~
    git clone https://github.com/PrasannaPulakurthi/MMD-AdversarialNAS.git
    cd MMD-AdversarialNAS
    ~~~
   
2. Install requirements using Python 3.9.

    ~~~
    conda create -n mmd-nas python=3.9
    conda activate mmd-nas
    pip install -r requirements.txt
    ~~~
    
2. Install PyTorch1 and Tensorflow2 with CUDA.

    ~~~
    pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
    ~~~
    To install other PyTorch versions compatible with your CUDA. [Install PyTorch](https://pytorch.org/get-started/previous-versions/)
   
    [Install Tensorflow](https://www.tensorflow.org/install/pip#windows-native)



## Instructions for Testing, Training, Searching, and Compressing the Model.   
### Preparing necessary files

Files can be found in [Google Drive](https://drive.google.com/drive/folders/1tcMf8Bj6m3iqh4UO-zGJbI_UdvYGMFWT?usp=sharing).

1. Download the pre-trained models to ./exps folder found [here](https://drive.google.com/drive/folders/1_-ymQyxItLkqSvoVuOZKSlnowCL4H6jz?usp=sharing).
    
2. Download the pre-calculated statistics to ./fid_stat for calculating the FID from [here](https://drive.google.com/drive/folders/1g0g-yZWAVPZJxyua0qWO25-O4zjhY15A?usp=sharing).

### Testing
1. Download the trained generative models from [here](https://drive.google.com/drive/folders/1_-ymQyxItLkqSvoVuOZKSlnowCL4H6jz?usp=sharing) to ./exps/arch_train_cifar10_large/Model

    ~~~
    mkdir -p exps/arch_train_cifar10_large/Model
    ~~~
   
2. To test the trained model, run the command found in scripts/test_arch_cifar10.sh
   
    ~~~
    python MGPU_test_arch.py --gpu_ids 0 --num_workers 8 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint arch_train_cifar10_large --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name arch_test_cifar10_large
    ~~~

### Training
1. Train the weights of the generative model with the searched architecture (the architecture is saved in ./exps/arch_cifar10/Genotypes/latest_G.npy). Run the command found in scripts/train_arch_cifar10_large.sh
   
    ~~~
    python MGPU_train_arch.py --gpu_ids 0 --num_workers 8 --gen_bs 128 --dis_bs 128 --dataset cifar10 --bottom_width 4 --img_size 32 --max_epoch_G 500 --n_critic 1 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --df_dim 512 --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --val_freq 5 --num_eval_imgs 50000 --exp_name arch_train_cifar10_large
    ~~~

### Searching the Architecture

1. To use AdversarialNAS to search for the best architecture, run the command found in scripts/search_arch_cifar10.sh
   
    ~~~
    python MGPU_search_arch.py --gpu_ids 0 --gen_bs 128 --dis_bs 128 --dataset cifar10 --bottom_width 4 --img_size 32 --max_epoch_G 25 --arch search_both_cifar10 --latent_dim 120 --gf_dim 160 --df_dim 80 --g_spectral_norm False --d_spectral_norm True --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --n_critic 5 --val_freq 5 --derive_freq 1 --derive_per_epoch 16 --draw_arch False --exp_name search/bs120-dim160 --num_workers 8 --gumbel_softmax True
    ~~~
    
### Compression
1. Compress and Finetune all the Convolutional Layers except 9 and 13.
   
    ~~~
    python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 8 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 100 --checkpoint arch_train_cifar10_large  --exp_name compress_train_cifar10_large --val_freq 5  --gen_bs  128 --dis_bs 128 --beta1 0.0 --beta2 0.9  --byrank --rank 256 --layers cell1.c0.ops.0.op.1 cell1.c1.ops.0.op.1 cell1.c2.ops.0.op.1 cell1.c3.ops.0.op.1 cell2.c0.ops.0.op.1 cell2.c2.ops.0.op.1 cell2.c3.ops.0.op.1 cell2.c4.ops.0.op.1 cell3.c1.ops.0.op.1 cell3.c2.ops.0.op.1 cell3.c3.ops.0.op.1 --compress-mode "allatonce" --max_epoch_G 500 --eval_before_compression
    ~~~
    
2. Compress the Fully Connected Layers except l1.

    ~~~
    python MGPU_cpcompress_arch.py --gpu_ids 0 --num_workers 8 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --genotypes_exp arch_cifar10  --latent_dim 120 --gf_dim 256 --df_dim 512 --num_eval_imgs 50000 --eval_batch_size 100 --checkpoint compress_train_cifar10_large  --exp_name compress_train_cifar10_large --val_freq 5  --gen_bs  128 --dis_bs 128 --beta1 0.0 --beta2 0.9  --byrank --rank 4 --layers l2 l3 --freeze_layers l2 l3 --compress-mode "allatonce" --max_epoch_G 1 --eval_before_compression
    ~~~
       
5. To Test the compressed network, download the compressed model from [Google Drive](https://drive.google.com/drive/folders/1xB6Y-btreBtyVZ-kdGTIZgLTjsv7H4Pd?usp=sharing) to ./exps/compress_train_cifar10_large/Model

    ~~~
    python MGPU_test_cpcompress.py --gpu_ids 0 --num_workers 8 --dataset cifar10 --bottom_width 4 --img_size 32 --arch arch_cifar10 --draw_arch False --checkpoint compress_train_cifar10_large --genotypes_exp arch_cifar10 --latent_dim 120 --gf_dim 256 --num_eval_imgs 50000 --eval_batch_size 100 --exp_name compress_test_cifar10_large  --byrank
    ~~~

## Citation
Please consider citing our paper in your publications if it helps your research. The following is a BibTeX reference.
```bibtex
@INPROCEEDINGS{10446488,
  author={Pulakurthi, Prasanna Reddy and Mozaffari, Mahsa and Dianat, Sohail A. and Rabbani, Majid and Heard, Jamison and Rao, Raghuveer},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Enhancing GAN Performance Through Neural Architecture Search and Tensor Decomposition}, 
  year={2024},
  volume={},
  number={},
  pages={7280-7284},
  keywords={Training;Performance evaluation;Tensors;Image coding;Image synthesis;Image edge detection;Computer architecture;Neural Architecture Search;Maximum Mean Discrepancy;Generative Adversarial Networks},
  doi={10.1109/ICASSP48485.2024.10446488}
}
```

## Acknowledgement
Codebase from [AdversarialNAS](https://github.com/chengaopro/AdversarialNAS), [TransGAN](https://github.com/VITA-Group/TransGAN), and [Tensorly](https://github.com/tensorly/tensorly).
