import os
import random


data_folder_path = "/Datasets/imagenet/imagenet21k_resized/imagenet21k_train"
save_path = "/scratch/ssolit/data/imagenet21k.txt"



# create shuffled list of all imagenet files
class_list = os.listdir(data_folder_path)
print('creating file_list', flush=True)
file_list = []
i = 0
for class_folder in class_list:
  class_files = os.listdir(data_folder_path + '/' + class_folder)
  file_list.extend(class_files)
  i+=1
  if (i%500 ==0):
    print(i, flush=True)

print('shuffling', flush=True)
random.shuffle(file_list)


# write shuffled filed to save_path
print('writing to ' + save_path, flush=True)
f = open(save_path, 'a')
for file_name in file_list:
  class_name = file_name.split("_")[0]
  og_path = data_folder_path + "/" + class_name + "/" + file_name
  assert(os.path.isfile(og_path))
  str = og_path + "\n"
  f.write(str)
  if (i%10000==0):
    print(i, flush=True)


print('finished, saved to ' + save_path)
