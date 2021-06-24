import os


desired_images = 100
imagenet_path = "/mnt/Data/Streaming_Data/imagenet/imagenet21k_resized/imagenet21k_train/"
image_folder = "/mnt/Data/Streaming_Data/imagenet/imagenet_512/images/"
save_path = "/home/ssolit/StreamingPerception/util_tools/test_txt.txt"

file_list = os.listdir(image_folder)

f = open(save_path, "w")

for file_name in file_list:
  class_name = file_name.split("_")[0]
  og_path = imagenet_path + "/" + class_name + "/" + file_name
  f.write(og_path)

f.close()
