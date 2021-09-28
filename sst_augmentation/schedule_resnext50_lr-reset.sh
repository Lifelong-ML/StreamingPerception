#!/bin/bash

sbatch --mem-per-gpu=16G --cpus-per-gpu=4 --gpus=1 --time=04:00:00 --qos=low --partition=compute python main_gft_correct_fc.py /scratch/ssolit/102flowers --classes 102 --a resnext50_32x4d --epochs 3000 --step 1200 --ckpt_dir /scratch/ssolit/StreamingPerception/resnext50_lr-reset/p1-reset/finetune --finetuned_model srun --mem-per-gpu=16G --cpus-per-gpu=4 --gpus=1 --time=04:00:00 --qos=low --partition=compute python main_gft_correct_fc.py /scratch/ssolit/102flowers --classes 102 --a resnext50_32x4d --epochs 3000 --step 1200 --ckpt_dir /scratch/ssolit/StreamingPerception/resnext50_lr-reset/p1-reset/finetune --finetuned_model /scratch/ssolit/StreamingPerception/resnext50_lr-reset/p1-reset/finetune/resnext50_32x4d_finetuned/checkpoint.state -b 64 --lr 0.025



sbatch --mem-per-gpu=16G --cpus-per-gpu=4 --gpus=1 --time=04:00:00 --qos=low --partition=compute python main_gft_correct_fc.py /scratch/ssolit/102flowers --classes 102 --a resnext50_32x4d --epochs 3000 --step 1200 --ckpt_dir /scratch/ssolit/StreamingPerception/resnext50_lr-reset/p1-reset/finetune --finetuned_model /scratch/ssolit/StreamingPerception/resnext50_lr-reset/p1-reset/finetune/resnext50_32x4d_finetuned/checkpoint.state -b 64 --lr 0.025

