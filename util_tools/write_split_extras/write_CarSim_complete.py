import os
import random


data_folder_path = '/scratch/ssolit/data/CarSim/complete/IMG'
save_path = "/scratch/ssolit/data/CarSim_txts/CarSim_complete.txt"



# create shuffled list of all files
print('creating file_list', flush=True)
file_list = os.listdir(data_folder_path)

# write shuffled filed to save_path
print('writing to ' + save_path, flush=True)
f = open(save_path, 'a')
i=0
for file_name in file_list:
  og_path = data_folder_path + "/" + file_name
  assert(os.path.isfile(og_path))
  str = og_path + "\n"
  f.write(str)
  if (i%10000==0):
    print(i, flush=True)
  i+=1

print('finished, saved to ' + save_path)
