print('first line print')
import os
from shutil import copyfile


data_dir = '/scratch/ssolit/data/ARID_40k_crop'
save_dir = '/scratch/ssolit/data/ARID_40k_crop_fewshot20val'
train_size = 10
val_size = 20
test_size = 200



#class_list = os.listdir(data_folder_path)

os.mkdir(save_dir)
os.mkdir(os.path.join(save_dir, 'train'))
os.mkdir(os.path.join(save_dir, 'val'))
os.mkdir(os.path.join(save_dir, 'test'))


j = 1
for class_type in sorted(os.listdir(data_dir)):
    print('starting class', j, class_type)

    class_dir = os.path.join(data_dir, class_type)
    class_splits = sorted(os.listdir(class_dir))

    split0_dir = os.path.join(class_dir, class_splits[0])
    split1_dir = os.path.join(class_dir, class_splits[1])

    split0_files = os.listdir(split0_dir)
    split1_files = os.listdir(split1_dir)


    fewshot_train_files = split0_files[0:train_size]
    fewshot_val_files = split0_files[train_size : train_size + val_size]
    fewshot_test_files = split1_files[0:test_size]


    os.mkdir(os.path.join(save_dir, 'train', class_type))
    os.mkdir(os.path.join(save_dir, 'val', class_type))
    os.mkdir(os.path.join(save_dir, 'test', class_type))

    for file in fewshot_train_files:
        copyfile(os.path.join(split0_dir, file), os.path.join(save_dir, 'train', class_type, file))

    for file in fewshot_val_files:
        copyfile(os.path.join(split0_dir, file), os.path.join(save_dir, 'val', class_type, file))

    for file in fewshot_test_files:
        copyfile(os.path.join(split1_dir, file), os.path.join(save_dir, 'test', class_type, file))

    j+=1





