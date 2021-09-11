import os
import random


data_folder_path = '/scratch/ssolit/data/Flowers299/'
save_path = '/scratch/ssolit/data/Flowers299_complete.txt'



# create shuffled list of all imagenet files
class_list = os.listdir(data_folder_path)
print('creating file_list', flush=True)
file_list = []
for class_folder in class_list:
  class_files = os.listdir(data_folder_path + '/' + class_folder)
  for file in class_files:
    file_list.append(class_folder + '/' + file)

print('shuffling', flush=True)
random.shuffle(file_list)


# write shuffled files to save_path
print('writing to ' + save_path, flush=True)
f = open(save_path, 'a')
i = 0
for file_name in file_list:
  og_path = data_folder_path + "/" + file_name
  assert(os.path.isfile(og_path))
  str = og_path + "\n"
  f.write(str)
  if (i%10000==0):
    print(i, flush=True)
  i+=1

print('finished, saved to ' + save_path)
