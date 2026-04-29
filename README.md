README.txt

Project Title

Conditional Diffusion Model with Guidance Scale Sweep (CIFAR-10)


1. Environment

This project was developed and executed on:

* ASU Sol Supercomputer (GPU environment)
* Python 3.10+
* PyTorch
* Torch Distributed (DDP)
* CUDA-enabled GPU

Required packages are listed in:

requirements.txt

Environment setup example:

conda create -n diffusion python=3.10
conda activate diffusion
pip install -r requirements.txt


2. Dataset

Dataset used:

CIFAR-10

Data preprocessing includes:

* Conversion to tensor
* Normalization to range [-1, 1]

Implemented in:

dataloader_cifar.py


3. Model Architecture

Model:

* U-Net based diffusion model
* Conditional embedding for class labels
* Gaussian diffusion process
* Classifier-Free Guidance (CFG)

Core implementation files:

unet.py
diffusion.py
embedding.py
Scheduler.py
utils.py


4. Training

Training script:

train.py

Example training command:

torchrun --nproc_per_node=1 train.py \
--epoch 300 \
--batchsize 64 \
--w 2.2 \
--threshold 0.05 \
--interval 25 \
--moddir model_300_w22_thr005 \
--samdir sample_300_w22_thr005

Key hyperparameters:

* Epochs: 300
* Batch size: 64
* Image size: 32x32
* Guidance scale (w): configurable
* Diffusion timesteps: default from scheduler


5. Sampling for FID

Sampling script:

generate_fid.py

Example command:

python generate_fid.py \
  --ckpt model_300_w22_thr005/ckpt_300_checkpoint.pt \
  --outdir fid_300_ckpt22_w22_10000 \
  --num_samples 10000 \
  --w 2.2

This generates PNG images for FID evaluation.


6. FID Calculation

FID was computed using:

pytorch-fid

Example command:

python -m pytorch_fid cifar_images fid_300_ckpt22_w22_10000

Reference images:

CIFAR-10 test set exported as PNG.

7. Inception Score (IS)

IS was computed using:

torchmetrics (InceptionScore)

Evaluation performed on generated image folders.

8. Experimental Design

We performed a sweep over guidance scale (ω):

* 1.8
* 2.0
* 2.2
* 2.5
* 2.7
* 3.0

Two experimental settings were evaluated:

1. Fixed checkpoint (w=2.5) with inference sweep
2. Fixed checkpoint (w=2.2) with inference sweep

Each configuration generated 10,000 samples for evaluation.


9. Notes

* Distributed Data Parallel (DDP) was used for GPU execution.
* Sampling and training were executed on Sol HPC.
* Evaluation (FID/IS) was performed using generated PNG outputs.

