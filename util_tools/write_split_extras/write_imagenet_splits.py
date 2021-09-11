import os
import random


imagenet_path = "/Datasets/imagenet/imagenet21k_resized/imagenet21k_train"
image_folder = "/Datasets/imagenet/imagenet21k_resized/imagenet21k_train"
save_path_1 = '/scratch/ssolit/data/rectangles/imagenet1M-7.txt'
save_path_2 = '/scratch/ssolit/data/imagenet_fixed1M/imagenet1M-8.txt'
save_path_3 = '/scratch/ssolit/data/imagenet_fixed1M/imagenet1M-9.txt'
#save_path_4 = '/scratch/ssolit/data/imagenet_fixed1M/imagenet1M-1.txt'
#save_path_5 = '/scratch/ssolit/data/imagenet_fixed1M/imagenet1M-.txt'

print('making orig list', flush=True)
class_list = os.listdir(image_folder)
file_list = []
for cl in class_list:
  file_list.append(os.listdir(image_folder + '/' + cl))
print('list length:', len(file_list))
print('shuffling', flush=True)
random.shuffle(file_list)


f1 = open(save_path_1, "a")
f2 = open(save_path_2, "a")
f3 = open(save_path_3, "a")
#f4 = open(save_path_4, "a")
#f5 = open(save_path_5, "a")

for i in range(0, 1000000*3):
  #create str that is the path to the relivent file
  file_name = file_list[i]
  class_name = file_name.split("_")[0]
  og_path = imagenet_path + "/" + class_name + "/" + file_name
  assert(os.path.isfile(og_path))
  str = og_path + "\n"

  #write str to file
  if (i < 1000000*1):
    f1.write(str)
  elif (i < 1000000*2):
    f2.write(str)
  elif (i < 1000000*3):
    f3.write(str)
  elif (i < 86400*4):
    f4.write(str)
  else:
    f5.write(str)


  if (i%10000==0):
    print(i, flush=True)
   

f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
