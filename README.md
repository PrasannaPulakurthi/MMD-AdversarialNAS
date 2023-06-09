# AdversarialNAS-MMD


## Getting Started
### Installation
1. Clone this repository.

    ~~~
    git clone https://github.com/chengaopro/AdversarialNAS.git
    ~~~
   
2. Install pytorch 1.1.0, tensorflow 1.9.0, CUDA 9.0 and corresponding CUDNN via conda.

    ~~~
    conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
    ~~~
   
    ~~~
    conda install tensorflow-gpu==1.9.0 cudnn
    ~~~
   
3. Install the requirements via pip.
    
    ~~~
    pip install -r requirements.txt
    ~~~
    
### Preparing necessary files

Files can be found in [Google Drive](https://drive.google.com/drive/folders/1xB6Y-btreBtyVZ-kdGTIZgLTjsv7H4Pd?usp=sharing).
   
2. Download the pre-calculated statistics to ./fid_stat for calculating the FID.
    
    ~~~
    mkdir fid_stat
    ~~~
   
3. Download the inception model to ./tmp for calculating the IS and FID.
    
    ~~~
    mkdir tmp
    ~~~
   
