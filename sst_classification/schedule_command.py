import time
#import subprocess
import os

just_print = False

'''
command_1 = 'srun --mem-per-gpu=16G --cpus-per-gpu=4 --gpus=1 --time=04:00:00 --qos=low --partition=compute python main_gft_correct_fc.py /scratch/ssolit/102flowers --classes 102 --a resnext50_32x4d --epochs 3000 --step 1200 --ckpt_dir /scratch/ssolit/StreamingPerception/resnext50_lr-reset/p1-reset/finetune --finetuned_model srun --mem-per-gpu=16G --cpus-per-gpu=4 --gpus=1 --time=04:00:00 --qos=low --partition=compute python main_gft_correct_fc.py /scratch/ssolit/102flowers --classes 102 --a resnext50_32x4d --epochs 3000 --step 1200 --ckpt_dir /scratch/ssolit/StreamingPerception/resnext50_lr-reset/p1-reset/finetune --finetuned_model /scratch/ssolit/StreamingPerception/resnext50_lr-reset/p1-reset/finetune/resnext50_32x4d_finetuned/checkpoint.state -b 64 --lr 0.025'
command_2 = 'srun --mem-per-gpu=16G --cpus-per-gpu=4 --gpus=1 --time=04:00:00 --qos=low --partition=compute python main_gft_correct_fc.py /scratch/ssolit/102flowers --classes 102 --a resnext50_32x4d --epochs 3000 --step 1200 --ckpt_dir /scratch/ssolit/StreamingPerception/resnext50_lr-reset/p1-reset/finetune --finetuned_model /scratch/ssolit/StreamingPerception/resnext50_lr-reset/p1-reset/finetune/resnext50_32x4d_finetuned/checkpoint.state -b 64 --lr 0.025'
'''


command_1 = 'srun --ntasks=1 --cpus-per-task=1 --time=04:00:00 --qos=low --partition=compute python main_gft_correct_fc.py /scratch/ssolit/102flowers --classes 102 --a resnext50_32x4d --epochs 3000 --step 1200 --ckpt_dir /scratch/ssolit/StreamingPerception/resnext50_lr-reset/p1-reset/finetune --finetuned_model srun --mem-per-gpu=16G --cpus-per-gpu=4 --gpus=1 --time=04:00:00 --qos=low --partition=compute python main_gft_correct_fc.py /scratch/ssolit/102flowers --classes 102 --a resnext50_32x4d --epochs 3000 --step 1200 --ckpt_dir /scratch/ssolit/StreamingPerception/resnext50_lr-reset/p1-reset/finetune --finetuned_model /scratch/ssolit/StreamingPerception/resnext50_lr-reset/p1-reset/finetune/resnext50_32x4d_finetuned/checkpoint.state -b 64 --lr 0.025'
command_2 = 'srun --ntasks=1 --cpus-per-task=1 --time=04:00:00 --qos=low --partition=compute python main_gft_correct_fc.py /scratch/ssolit/102flowers --classes 102 --a resnext50_32x4d --epochs 3000 --step 1200 --ckpt_dir /scratch/ssolit/StreamingPerception/resnext50_lr-reset/p1-reset/finetune --finetuned_model /scratch/ssolit/StreamingPerception/resnext50_lr-reset/p1-reset/finetune/resnext50_32x4d_finetuned/checkpoint.state -b 64 --lr 0.025'





command_list = [command_1, command_2]
for i in range(1, len(command_list)):
    begin_str = ' --begin=now+' + str(4*i) + 'hour'
    command_list[i] = command_list[i][0:4] + begin_str + command_list[i][4:]


print('commands to run:')
for command in command_list:
    print(command)
    print()
print()
print()


if just_print:
    exit()

for command in command_list:
    os.system(command)

'''
for command in command_list:
    print(command)
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print('start time:', current_time, flush=True)
    subprocess.run(command.split(' '), capture_output=True)
    exit()

    print('finsihed ' + command)
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print('finish time:', current_time)
    print()
    print()
'''

print('finished run_command.py')

