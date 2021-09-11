import os
import random


data_folder_path = '/scratch/ssolit/data/noise/images'
sub_folders = False	#True is there are class folders inside data_foldes
save_path_1 = '/scratch/ssolit/data/noise/noise-1.txt'
save_path_2 = '/scratch/ssolit/data/noise/noise-2.txt'
save_path_3 = '/scratch/ssolit/data/noise/noise-3.txt'





# create shuffled list of all files in data_folder
print('creating file_list', flush=True)
file_list = []
if (sub_folders):
  class_list = os.listdir(data_folder_path)
  for class_folder in class_list:
    class_files = os.listdir(data_folder_path + '/' + class_folder)
    for file in class_files:
      file_list.append(class_folder + '/' + file)
else:
  file_list = os.listdir(data_folder_path)

print('shuffling', flush=True)
random.shuffle(file_list)


# write shuffled files to save_path
print('writing to ' + save_path_1, flush=True)
f1 = open(save_path_1, 'a')
f2 = open(save_path_2, 'a')
f3 = open(save_path_3, 'a')

i = 0
for file_name in file_list:
  og_path = data_folder_path + "/" + file_name
  assert(os.path.isfile(og_path))
  str = og_path + "\n"
  if (i < 30000):
    f1.write(str)
  elif (i < 60000):
    f2.write(str)
  elif (i < 90000):
    f3.write(str)

  if (i%10000==0):
    print(i, flush=True)
  i+=1

print('finished, saved to ' + save_path_1)
