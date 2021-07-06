'''
# choose parameters
server = "gc"                  #should be "lml" or "gc"
class_data = "flowers"
experiment = "tiny_scratch"
arch = "resnext50"
stage = 3
data_path = "/scratch/ssolit/102flowers"
stream_data_path = "/home/ssolit/imagenet3M/"
#resume = '/scratch/ssolit/StreamingPerception/f_1M_init_testing/f_1M_init_test4/resnet18/init/resnet18_scratch/model_best.state'
resume = None
data_txt = '/scratch/ssolit/StreamingPerception/tiny_scratch/resnext50/resnet18_scratch.txt'


# choose parameters
server = "gc"                  #should be "lml" or "gc"
class_data = "flowers"
experiment = "f_it4"
arch = "resnext50"
stage = 4
data_path = "/scratch/ssolit/102flowers"
stream_data_path = "/home/ssolit/imagenet3M/"
#resume = '/scratch/ssolit/StreamingPerception/f_1M_init_testing/f_1M_init_test4/resnet18/init/resnet18_scratch/model_best.state'
resume = None
data_txt = '/scratch/ssolit/StreamingPerception/f_it4/resnext50/resnet18_scratch.txt'
#log = '/scratch/ssolit/StreamingPerception/f_1M_init_test1/resnet18/init/train_log.txt'
'''

server = "gc"                  #should be "lml" or "gc"
class_data = "flowers"
experiment = "f_it4p3"
arch = "resnext101"
stage = 3
data_path = "/scratch/ssolit/102flowers"
stream_data_path = "/home/ssolit/imagenet3M/"
resume = '/scratch/ssolit/StreamingPerception/f_it4p3/resnext101/pseudo_train/resnext101_32x8d_scratch/checkpoint.state'
#resume = None
data_txt = '/scratch/ssolit/StreamingPerception/f_it4p3/resnext101/resnext50_32x4d_scratch.txt'
#log = '/scratch/ssolit/StreamingPerception/f_1M_init_test1/resnet18/init/train_log.txt'



import os
assert(os.path.isdir(data_path))
assert(os.path.isdir(stream_data_path))
if (resume != None):
    assert(os.path.isfile(resume)), (resume + ' is not a valid filepath')
if(data_txt != None):
    assert(os.path.isfile(data_txt))

mem_per_gpu = "64G"
cpus_per_gpu = "16"
gpus = "1"
time = "24:00:00"
qos = "eaton-high"
partition = "eaton-compute"



def get_slurm_str():
    slurm_str = "srun"
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
    else:
        print("Error: unrecognized class_data")
        exit(1)
    return class_num

def get_dir_path():
    path = ""
    if (server == "gc"):
        path += "/scratch/ssolit/StreamingPerception/" + experiment + "/" + arch
    elif (server == "lml"):
        path += "/mnt/Data/Streaming_Data/" + experiment + "/" + arch
    return path

def get_py_str():
    dir_path = get_dir_path()
    arch_name = arch
    if (arch == "resnext50"):
      arch_name = "resnext50_32x4d"
    elif (arch == 'resnext101'):
      arch_name = 'resnext101_32x8d'
    py_str = "python3"
    #initialize
    if (stage == 1):
        py_str += " main_correct_size.py"
        py_str += " " + data_path
        py_str += " --classes " + get_class_num()
        py_str += " --a " + arch_name
        py_str += " --epochs 3000"
        py_str += " --step 1200"
        py_str += " --ckpt_dir " + dir_path + "/init"
        if (resume!=None):
          py_str += " --resume " + resume
#        py_str += ' --log ' + log
    #Pseudolabel
    elif (stage == 2):
        py_str += " generate_labels_correct_fc.py"
        py_str += " " + stream_data_path
        py_str += " --classes " + get_class_num()
        py_str += " --a " + arch_name
        py_str += " --data_save_dir " + dir_path
        if (resume!=None):
          py_str += " --resume " + resume
        if(data_txt != None):
          py_str += " --data_txt " + data_txt
    #Pseudo Train
    elif (stage == 3):
        py_str += " main_g_correct_fc.py"
        py_str += " " + get_dir_path()
        py_str += " --classes " + get_class_num()
        py_str += " --a " + arch_name
        py_str += " --epochs 30"
        py_str += " --step 25"
        py_str += " --ckpt_dir " + dir_path + "/pseudo_train"
        if (resume!=None):
          py_str += " --resume " + resume
        if(data_txt != None):
          py_str += " --data_txt " + data_txt
    #Finetune
    elif (stage == 4):
        py_str += " main_gft_correct_fc.py"
        py_str += " " + data_path
        py_str += " --classes " + get_class_num()
        py_str += " --a " + arch_name
        py_str += " --epochs 3000"
        py_str += " --step 1200"
        py_str += " --ckpt_dir " + dir_path + "/finetune"
        py_str += " --finetuned_model " + dir_path + "/finetune/" + arch + "_finetuned/checkpoint.state"
    #Evaluate
    else:
        py_str += " main_gft_correct_fc.py"
        py_str += " " + data_path
        py_str += " --classes " + get_class_num()
        py_str += " --a " + arch_name
        py_str += " --epochs 3000"
        py_str += " --step 1200"
        py_str += " --ckpt_dir " + dir_path + "/finetune"
        py_str += " --evaluate"
        py_str += " --finetuned_model " + dir_path + "/finetune/" + arch_name + "_finetuned/model_best.state"

    if (server == "lml"):
        py_str += " -b 128"
    return py_str

final_str = ""
if (server == "gc"):
    final_str += get_slurm_str() + " "
final_str += get_py_str()
print(final_str)

