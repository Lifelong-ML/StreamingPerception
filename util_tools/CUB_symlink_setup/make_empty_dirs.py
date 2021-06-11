# importing os module
import os


base_cub_path = "/Datasets/CUB_200_2011/"


train_parent = "/scratch/ssolit/CUB_links/train"
val_parent = "/scratch/ssolit/CUB_links/val"

f = open(base_cub_path + "classes.txt", "r")
for line in f:
  line_list = line.split(" ")
  new_dir_name = line_list[1]
  
  train_path = str.rstrip(os.path.join(train_parent, "train." + new_dir_name))
  val_path = str.rstrip(os.path.join(val_parent, "val." + new_dir_name))
  print(train_path)
  print(val_path)

  os.mkdir(train_path)
  os.mkdir(val_path)
f.close()



# Source file path
#src = '/home/ihritik/file.txt'

# Destination file path
#dst = '/home/ihritik/Desktop/file(symlink).txt'

# Create a symbolic link
# pointing to src named dst
# using os.symlink() method
#os.symlink(src, dst)
