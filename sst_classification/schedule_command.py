import os
import subprocess
import requests
import json

eaton_compute = True



experiment = 'scratch_test/p1'
server = 'gc'
class_data = 'arid'
arch = 'resnet18'
optimizer = 'adam'
augment = False
eaton_compute = False
stage = 2
data_txt = '/scratch/ssolit/data/imagenet_100_pics.txt'
resume = '/scratch/ssolit/StreamingPerception/scratch_test/p0/init/resnet18_scratch/model_best.state'
#resume = None


'''
experiment = 'f_resnet18_lrReset/p4'
server = 'gc'
class_data = 'flowers'
arch = 'resnet18'
optimizer = 'svg'
augment = False
stage = 4
resume = '/scratch/ssolit/StreamingPerception/f_resnet18_lrReset/p4/ereset_model_best.state'
#resume = None
data_txt = '/scratch/ssolit/data/imagenet_137/imagenet7M.txt'
#eaton_compute = False
'''






def main():
    #print(get_py_str())
    schedule_cycle(start_stage=stage)




if (resume != None):
    assert(os.path.isfile(resume)), (resume + ' is not a valid filepath')
if(data_txt != None):
    assert(os.path.isfile(data_txt))



def schedule_cycle(start_stage=2):
    print()
    commands = []
    if (start_stage == 1):
        commands = [get_py_str(1, None, None)]
    else:
        for i in range(start_stage, 5):
            stage = i
            resume, data_txt = get_resume_datatxt(stage)
            commands.append(get_py_str(stage, resume, data_txt))
    for command in commands:
        print(command)
        try:
            subprocess.check_call(command.split(' '))
        except subprocess.CalledProcessError:
            error_message = "Job threw an error:\n\n" + command
            post_message_to_slack(error_message)
            exit(1)
    success_message = "job_cycle finished successfully. Experiment: " + experiment
    post_message_to_slack(success_message)

def get_resume_datatxt(stage):
    global resume
    global data_txt
    if (server == 'gc'):
       exp_prefix = '/scratch/ssolit/StreamingPerception/'
    else:
       print('unimplimented')
       exit(1)

    if (stage == 2):
        return resume, data_txt
    elif (stage == 3):
        data_txt = exp_prefix + experiment + '/' + get_arch_name() + '_scratch.txt'
        return None, data_txt
    elif (stage == 4):
        stage_3_checkpoint = exp_prefix + experiment + '/pseudo_train/resnet18_scratch/checkpoint.state'
        return stage_3_checkpoint, None
    else:
        print('unimplimented stage for get_resume_data')
        exit()

def post_message_to_slack(text, blocks = None):
    slack_token = 'xoxb-127660415360-2555645733079-DWb5lr9lZFbiD36UITCWiuEg'
    slack_channel = '@U01K43ABPMX'
    slack_icon_emoji = ':see_no_evil:'
    slack_user_name = 'Streaming Moniter'

    return requests.post('https://slack.com/api/chat.postMessage', {
        'token': slack_token,
        'channel': slack_channel,
        'text': text,
        'icon_emoji': slack_icon_emoji,
        'username': slack_user_name,
        'blocks': json.dumps(blocks) if blocks else None
    }).json()


#functions to command string
def get_slurm_str():
  if (eaton_compute):
    mem_per_gpu = "32G"
    cpus_per_gpu = "16"
    gpus = "1"
    time = "24:00:00"
    qos = "eaton-high"
    partition = "eaton-compute"
  else:
    mem_per_gpu = "16G"
    cpus_per_gpu = "4"
    gpus = "1"
    time = "04:00:00"
    qos = "low"
    partition = "compute"

    slurm_str = "srun"
    slurm_str += " --begin=now"

    cwd = os.getcwd()
    slurm_str += " --exclude " + cwd + "/1080s"

    slurm_str += " --mem-per-gpu=" + mem_per_gpu
    slurm_str += " --cpus-per-gpu=" + cpus_per_gpu
    slurm_str += " --gpus=" + gpus
    slurm_str += " --time=" + time
    slurm_str += " --qos=" + qos
    slurm_str += " --partition=" + partition
    return slurm_str

def get_class_num():
    class_num = 0
    if (class_data == "flowers"):
        class_num = "102"
    elif (class_data == "CUB"):
        class_num = "200"
    elif (class_data == "arid"):
        class_num = '51'
    elif (class_data == "cifar"):
        class_num = '100'
    else:
        print("Error: unrecognized class_data")
        exit(1)
    return class_num

def get_dir_path():
    path = ""
    if (server == "gc"):
        path += "/scratch/ssolit/StreamingPerception/" + experiment
    elif (server == "lml"):
        path += "/mnt/Data/Streaming_Data/" + experiment
    return path

def get_data_path():
    path = ""
    if (server == "gc"):
        if (class_data == "flowers"):
            path = "/scratch/ssolit/102flowers"
        elif (class_data == "CUB"):
            path = "/scratch/ssolit/CUB_links"
        elif (class_data == "arid"):
            path = "/scratch/ssolit/data/ARID_40k_crop_fewshot/"
        elif (class_data == "cifar"):
            path = "/scratch/ssolit/data/CIFAR100/fewshot_data/"
        else:
            print("Error: unrecognized class_data")
            exit(1)
    else:
        print("unimplemented")
        exit(1)
    return path

def get_batch_str():
  if (optimizer == 'svg'):
    lr = "0.025"
  elif (optimizer == 'adam'):
    lr = "0.001"
  else:
    print("Error: unrecognized class_data")
    exit(1)
  return " -b=64 --lr=" + lr

def get_arch_name():
    arch_name = arch
    if (arch == "resnext50"):
      arch_name = "resnext50_32x4d"
    elif (arch == 'resnext101'):
      arch_name = 'resnext101_32x8d'
    return arch_name

def get_py_str(stage, resume, data_txt):
    dir_path = get_dir_path()
    data_path = get_data_path()
    arch_name = get_arch_name()
    if (arch == "resnext50"):
      arch_name = "resnext50_32x4d"
    elif (arch == 'resnext101'):
      arch_name = 'resnext101_32x8d'

    py_str = ""
    if (server == "gc"):
        py_str += get_slurm_str() + " "
    py_str += "python"
    #initialize
    if (stage == 1):
        py_str += " main_correct_size.py"
        py_str += " " + data_path
        py_str += " --classes=" + get_class_num()
        py_str += " --arch=" + arch_name
        py_str += " --optimizer=" + optimizer
        py_str += " --epochs=3000"
        if (optimizer == 'svg'):
            py_str += " --step=1200"
        py_str += " --ckpt_dir=" + dir_path + "/init"
        py_str += get_batch_str()
        if (resume!=None):
          py_str += " --resume=" + resume
    #Pseudolabel
    elif (stage == 2):
        py_str += " generate_labels_correct_fc.py"
        py_str += " /scratch/ssolit/data/"
        py_str += " --classes=" + get_class_num()
        py_str += " --arch=" + arch_name
        py_str += " --data_save_dir=" + dir_path

        if(resume != None):
          py_str += " --resume=" + resume
        if(data_txt != None):
          py_str += " --data_txt=" + data_txt
        if(augment):
          py_str += " --aug=True"
    #Pseudo Train
    elif (stage == 3):
        py_str += " main_g_correct_fc.py"
        py_str += " " + get_dir_path()
        py_str += " --classes=" + get_class_num()
        py_str += " --arch=" + arch_name
        py_str += " --optimizer=" + optimizer
        py_str += " --epochs=3"
        if (optimizer == 'svg'):
            py_str += " --step=25"
        py_str += " --ckpt_dir=" + dir_path + "/pseudo_train"
        py_str += get_batch_str()

        if(data_txt != None):
          py_str += " --data_txt=" + data_txt
        if(augment):
          py_str += " --aug=True"
        future_resume = dir_path + "/pseudo_train/resnet18_scratch/checkpoint.state"
        if not os.path.isfile(future_resume):
            print('Warning: the following may be needed in future calls:')
            print('--resume ' + future_resume + '\n')
        else:
            py_str += " --resume=" + future_resume
    #Finetune
    elif (stage == 4):
        py_str += " main_gft_correct_fc.py"
        py_str += " " + data_path
        py_str += " --classes=" + get_class_num()
        py_str += " --arch=" + arch_name
        py_str += " --optimizer=" + optimizer
        py_str += " --epochs=50"
        if (optimizer == 'svg'):
            py_str += " --step=1200"
        py_str += " --ckpt_dir=" + dir_path + "/finetune"
        py_str += get_batch_str()
        future_finetuned_model = dir_path + "/finetune/" + arch_name + "_finetuned/checkpoint.state"
        if not os.path.isfile(future_finetuned_model):
            print('Warning: the following may be needed in future calls:')
        print('--finetuned_model ' + future_finetuned_model + '\n')
        py_str += " --finetuned_model=" + resume
    #Evaluate
    '''
    elif (stage == 5):
        py_str += " main_gft_correct_fc.py"
        py_str += " " + data_path
        py_str += " --classes " + get_class_num()
        py_str += " --a " + arch_name
        py_str += " --epochs 3000"
        py_str += " --step 1200"
        py_str += " --ckpt_dir " + dir_path + "/finetune"
        py_str += get_batch_str()
        py_str += " --evaluate"
        py_str += " --finetuned_model " + dir_path + "/finetune/" + arch_name + "_finetuned/model_best.state"
    '''
    return py_str


main()
