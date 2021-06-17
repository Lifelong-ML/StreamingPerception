import os
import shutil
import random as rand

print("starting")

imagenet_path = "/Datasets/imagenet/imagenet21k_resized/imagenet21k_train/"
desired_images = 3000000
save_name = "imagenet_3M"
save_path = "/scratch/ssolit/" + save_name + "/"



#make directory
'''
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)
'''

#randomly pick from imagenet
classes = os.listdir(imagenet_path)

i = len(os.listdir(save_path))
while(i < desired_images):
  im_index = rand.randint(0, len(classes) - 1)
  rand_class = classes[im_index]
  class_path = imagenet_path + rand_class + "/"
  del classes[im_index]

  class_images = os.listdir(class_path)
  rand_image = rand.choice(class_images)
  image_path = class_path + rand_image

  dst = save_path + rand_image
  if not os.path.isfile(dst):
    os.symlink(image_path, dst)
    i+=1
  if (i%5000==0):
    print(i)
