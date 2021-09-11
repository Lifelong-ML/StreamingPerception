# importing os module
import os
import glob

base_cub_path = "/Datasets/CUB_200_2011/"
train_parent = "/scratch/ssolit/CUB_links/train"
val_parent = "/scratch/ssolit/CUB_links/val"

# Read files and make arrays storing name and whether it is train/val
tt_split_list = []
f1 = open(base_cub_path + "train_test_split.txt", "r")
for line in f1:
  tt_split_list.append(str.rstrip(line.split(" ")[1]))
f1.close()

image_list = []
f2 = open(base_cub_path + "images.txt", "r")
for line in f2:
  image_list.append(str.rstrip(line.split(" ")[1]))
f2.close()

# remove links from folder that image shouldn't be in
for tt, img in zip(tt_split_list, image_list):
  print("tt = ", tt, ", img = ", img)
  train_file = train_parent + '/train.' + img
  val_file = val_parent + '/val.' + img

  if (tt == '0'):
    os.remove(train_file) #as <is_training_image> is false
  elif(tt == '1'):
    os.remove(val_file)


'''
for tt, img in zip(tt_split_list, image_list):
  print("tt = ", tt, ", img = ", img)
  try:
    if(tt == '0'):
      files=glob.glob(train_parent + "*/" + img)
      for file in files:
        print("0. file = ", file)
        os.unlink(file)
    else:
      files = glob.glob(val_parent + "*/" + img)
      for file in files:
        print("1. file = ", file)
        os.unlink(file)
  except OSError as e:
        print("Error: %s : %s" % (f, e.strerror))
'''

