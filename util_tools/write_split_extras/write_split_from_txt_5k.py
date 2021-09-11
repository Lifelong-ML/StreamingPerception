import os
import random


imagenet_path = "/Datasets/imagenet/imagenet21k_resized/imagenet21k_train"
txt_path = '/scratch/ssolit/data/imagenet21k.txt'
assert(os.path.isfile(txt_path))
save_path_1 = "/scratch/ssolit/data/imagenet5k/imagenet5k-1.txt"
save_path_2 = "/scratch/ssolit/data/imagenet5k/imagenet5k-2.txt"
save_path_3 = "/scratch/ssolit/data/imagenet5k/imagenet5k-3.txt"
save_path_4 = "/scratch/ssolit/data/imagenet5k/imagenet5k-4.txt"
save_path_5 = "/scratch/ssolit/data/imagenet5k/imagenet5k-5.txt"


print('making orig list', flush=True)
f = open(txt_path)
file_list = []
for line in f:
  file_list.append(line)
f.close()

print('list length:', len(file_list))
print('shuffling', flush=True)
random.shuffle(file_list)



f1 = open(save_path_1, "a")
f2 = open(save_path_2, "a")
f3 = open(save_path_3, "a")
f4 = open(save_path_4, "a")
f5 = open(save_path_5, "a")


print('writing to files', flush=True)
i = 0
for file_str in file_list:
  #assert os.path.isfile(file_str.strip('\n')), file_str.strip('\n') + "is not a valid path"

  #write str to file
  if (i < 5000*1):
    f1.write(file_str)
  elif (i < 5000*2):
    f2.write(file_str)
  elif (i < 5000*3):
    f3.write(file_str)
  elif (i < 5000*4):
    f4.write(file_str)
  elif (i < 5000*5):
    f5.write(file_str)
  else:
    break

  if (i%10000==0):
    print(i, flush=True)
  i+=1


f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
