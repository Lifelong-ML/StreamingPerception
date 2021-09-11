import os
import random


txt_path = '/scratch/ssolit/data/CarSim_txts/CarSim_complete.txt'
save_path_1 = '/scratch/ssolit/data/CarSim_txts/CarSim-30k-1.txt'
#save_path_2 = "/scratch/ssolit/data/imagenet_137/imagenet7M.txt"
#save_path_3 = "/scratch/ssolit/data/imagenet_big/imagenet7M.txt"
assert(os.path.isfile(txt_path))



print('making orig list', flush=True)
f = open(txt_path)
file_list = []
for line in f:
  file_list.append(line)
f.close()

print('list length:', len(file_list))
print('shuffling', flush=True)
random.shuffle(file_list)



print('writing to new files', flush=True)
f1 = open(save_path_1, "a")
#f2 = open(save_path_2, "a")
#f3 = open(save_path_3, "a")

i = 0
for line in file_list:
  #create str that is the path to the relivent file
  str = line

  #write str to file
  if (i < 30000):
    f1.write(str)
  else:
    break
  '''
  elif (i < 10000000):
    f2.write(str)
  else:
    f3.write(str)
  '''
  i+=1
  if (i%100000==0):
    print(i, flush=True)


f1.close()
#f2.close()
#f3.close()

