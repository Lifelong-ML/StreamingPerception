# choose parameters
server = "lml"                  #should be "lml" or "gc"
class_data = "flowers"
experiment = "flowers1"
arch = "resnet50"
stage = 1
data_path = "/mnt/Data/Streaming_Data/flower_data"


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
        path += "/scratch/ssolit/StreamingPerception/" + experiment
    elif (server == "lml"):
        path += "../" + experiment
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
        py_str += " --classes " + get_class_num()
        py_str += " --a " + arch
        py_str += " --resume " + get_dir_path() + "/init/resnet18_scratch/model_best.pth.tar"
        py_str += " --data_save_dir " + get_dir_path()
    #Pseudo Train
    elif (stage == 3):
        py_str += " main_g_correct_fc.py"
        py_str += " " + get_dir_path()
        py_str += " --classes " + get_class_num()
        py_str += " --a " + arch
        py_str += " --epochs 30"
        py_str += " --step 25"
        py_str += " --ckpt_dir " + get_dir_path() + "/pseudo_train"
        py_str += " --resume " + get_dir_path() + "/pseudo_train/resnet18_scratch/checkpoint.pth.tar"
    #Finetune
    elif (stage == 4):
        py_str += " main_gft_correct_fc.py"
        py_str += " " + data_path
        py_str += " --classes " + get_class_num()
        py_str += " --a " + arch
        py_str += " --epochs 4000"
        py_str += " --step 25"
        py_str += " --ckpt_dir " + get_dir_path() + "/finetune"
        py_str += " --finetuned_model " + get_dir_path() + "/finetune/resnet18_scratch/checkpoint.pth.tar"
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
        py_str += " --finetuned_model " + get_dir_path() + "/finetune/resnet18_finetuned/model_best.pth.tar"
    return py_str

final_str = ""
if (server == "gc"):
    final_str += get_slurm_str() + " "
final_str += get_py_str()
print(final_str)

