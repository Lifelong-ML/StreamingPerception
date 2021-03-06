import os
import shutil
import random as rand

print("starting", flush=True)

imagenet_path = "/Datasets/imagenet/imagenet21k_resized/imagenet21k_train/"
desired_images = 3000000
save_name = "imagenet3M"
save_path = "/home/ssolit/" + save_name + "/"

assert(os.path.isdir(save_path)), (save_path + " failed")

#delete directory and make new directory
'''
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)
'''

print("making im list", flush=True)
#make a list of every image
classes = os.listdir(imagenet_path)
im_list = []
for im_class in classes:
  images = os.listdir(imagenet_path + im_class)
  for image in images:
    im_list.append([im_class, image])
print("finished im list", flush=True)


#randomly pick from imagenet
i = len(os.listdir(save_path))
while(i < desired_images):
  index = rand.randint(0, len(im_list) - 1)
  im = im_list.pop(index)
  src = imagenet_path + im[0] + "/" + im[1]
  dst = save_path + im[1]
  if not os.path.isfile(dst):
    assert(os.path.isfile(src))
    os.symlink(src, dst)
    i+=1
  if (i%5000==0):
    print(i, flush=True)
