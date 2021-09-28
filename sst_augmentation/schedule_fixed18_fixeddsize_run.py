#!/bin/bash


#SBATCH --mem-per-gpu=16G
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --qos=low
#SBATCH --partition=compute
#Sbatch --job-name=fixed18_fixeddsize_P4S3

python main_g_correct_fc.py /scratch/ssolit/StreamingPerception/fixed18_fixeddsize/p4 --classes 102 --a resnet18 --epochs 30 --step 25 --ckpt_dir /scratch/ssolit/StreamingPerception/fixed18_fixeddsize/p4/pseudo_train --data_txt /scratch/ssolit/StreamingPerception/fixed18_fixeddsize/p4/resnet18_scratch.txt
wait
python main_g_correct_fc.py /scratch/ssolit/StreamingPerception/fixed18_fixeddsize/p4 --classes 102 --a resnet18 --epochs 30 --step 25 --ckpt_dir /scratch/ssolit/StreamingPerception/fixed18_fixeddsize/p4/pseudo_train --data_txt /scratch/ssolit/StreamingPerception/fixed18_fixeddsize/p4/resnet18_scratch.txt --resume /scratch/ssolit/StreamingPerception/fixed18_fixeddsize/p4/pseudo_train/resnet18_scratch/checkpoint.state


