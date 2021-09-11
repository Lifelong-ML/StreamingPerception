# importing os module
import os

base_cub_path = "/Datasets/CUB_200_2011/"
link_folder = "/scratch/ssolit/CUB_links"

#make empty folders

if not os.path.isdir(link_folder):
  os.mkdir(link_folder)
train_folder = link_folder + "/train"
val_folder = link_folder + "/val"
os.mkdir(train_folder)
os.mkdir(val_folder)

f = open(base_cub_path + "classes.txt", "r")
for line in f:
  line_list = line.split(" ")
  new_dir_name = line_list[1]

  train_class = str.rstrip(os.path.join(train_folder, new_dir_name))
  val_class = str.rstrip(os.path.join(val_folder, new_dir_name))
  print(train_class)
  print(val_class)

  os.mkdir(train_class)
  os.mkdir(val_class)
f.close()

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
  base_file = base_cub_path + "images/" + img
  train_file = train_folder + '/' + img
  val_file = val_folder + '/' + img
  assert os.path.isfile(base_file), base_file + " does not exist"


  if (tt == '0'):		# meaning <is_training_image> is false
    os.symlink(base_file, val_file)
  elif(tt == '1'):
    os.symlink(base_file, train_file)




