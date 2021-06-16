# choose parameters
server = "lml"                  #should be "lml" or "gc"
class_data = "flowers"
experiment = "flowers2"
arch = "resnet50"
stage = 5
data_path = "/mnt/Data/Streaming_Data/102flowers/"
stream_data_path = "/mnt/Data/Streaming_Data/imagenet/tiny-imagenet-200"
#resume = "/mnt/Data/Streaming_Data/flowers2/resnet18/finetune/resnet18_finetuned/model_best.state"
resume = None

'''
Commone data_path options:
/mnt/Data/Streaming_Data/flowers
'''

mem_per_gpu = "32G"
cpus_per_gpu = "16"
gpus = "1"
time = "6:00:00"
qos = "eaton-high"
partition = "-eaton-compute"



def get_slurm_str():
    slurm_str = "srun"
    slurm_str += " --mem-per-gpu=" + mem_per_gpu
    slurm_str += " --cpus-per-gpu=" + cpus_per_gpu
    slurm_str += " --gpus=" + gpus
    slurm_str += "--time=" + time
    slurm_str += " --qos=" + qos
    slurm_str += " -partition=" + partition
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
    py_str = "python3"
    #initialize
    if (stage == 1):
        py_str += " main_correct_size.py"
        py_str += " " + data_path
        py_str += " --classes " + get_class_num()
        py_str += " --a " + arch
        py_str += " --epochs 4000"
        py_str += " --step 1500"
        py_str += " --ckpt_dir " + get_dir_path() + "/init"
    #Pseudolabel
    elif (stage == 2):
        py_str += " generate_labels_correct_fc.py"
        py_str += " " + stream_data_path
        py_str += " --classes " + get_class_num()
        py_str += " --a " + arch
        py_str += " --data_save_dir " + get_dir_path()
        if (resume!=None):
          py_str += " --resume " + resume
    #Pseudo Train
    elif (stage == 3):
        py_str += " main_g_correct_fc.py"
        py_str += " " + get_dir_path()
        py_str += " --classes " + get_class_num()
        py_str += " --a " + arch
        py_str += " --epochs 30"
        py_str += " --step 25"
        py_str += " --ckpt_dir " + get_dir_path() + "/pseudo_train"
        if (resume!=None):
          py_str += " --resume " + resume
    #Finetune
    elif (stage == 4):
        py_str += " main_gft_correct_fc.py"
        py_str += " " + data_path
        py_str += " --classes " + get_class_num()
        py_str += " --a " + arch
        py_str += " --epochs 4000"
        py_str += " --step 25"
        py_str += " --ckpt_dir " + get_dir_path() + "/finetune"
        py_str += " --finetuned_model " + get_dir_path() + "/finetune/" + arch + "_finetuned/checkpoint.state"
    #Evaluate
    else:
        py_str += " main_gft_correct_fc.py"
        py_str += " " + data_path
        py_str += " --classes " + get_class_num()
        py_str += " --a " + arch
        py_str += " --epochs 4000"
        py_str += " --step 25"
        py_str += " --ckpt_dir " + get_dir_path() + "/finetune"
        py_str += " --evaluate"
        py_str += " --finetuned_model " + get_dir_path() + "/finetune/resnet18_finetuned/model_best.state"
    if (server == "lml"):
        py_str += " -b 128"
    return py_str

final_str = ""
if (server == "gc"):
    final_str += get_slurm_str() + " "
final_str += get_py_str()
print(final_str)

