print('first line print')

import os
import cv2
from albumentations.augmentations.geometric.transforms import ShiftScaleRotate
from albumentations.augmentations.transforms import ColorJitter, Superpixels, ToGray, GaussNoise, GaussianBlur


'''
og_dataset_txt = ''
save_directory = ''

if not os.pasth.isdir(save_folder):
  os.mkdir(save_folder)
img_directory = save_folder + '/imgs'
if not os.path.isdir(img_directory):
  os.mkdir(img_directory)

f = open(og_dataset_txt)
file_list = []
for line in f:
  file_list.append(line.strip('\n'))
f.close()
assert(os.path.isfile(file_list[0]))
'''

img_directory = '/scratch/ssolit/data/aug_test'
if not os.path.isdir(img_directory):
  os.mkdir(img_directory)
file_list = ['/scratch/ssolit/data/Flowers299/Abutilon/ea26410a5f.jpg', '/scratch/ssolit/data/BikeVideo/2806/frame_20340.jpg']
assert(os.path.isfile(file_list[0]))


# define transforms
rotate = ShiftScaleRotate(p=1)
jitter = ColorJitter(p=1)
supPix = Superpixels(p=1)
toGray = ToGray(p=1)
gaussNoise = GaussNoise(p=1)
gaussBlur = GaussianBlur(p=1)


# create augmented images
for img_path in file_list:
  image = cv2.imread(img_path)
  transformed = [image]

  print('applying geometric transforms')
  for i in range(3):
    transformed.append(rotate(image=image)['image'])

  print('applying noise and blur')
  for i in range(4):
    transformed.append(gaussNoise(image=transformed[i])['image'])
#    transformed.append(gaussBlur(image=transformed[i])['image'])

  print('applying jitter and grey')
  for i in range(8):
    transformed.append(jitter(image=transformed[i])['image'])
    transformed.append(toGray(image=transformed[i])['image'])
    transformed.append(supPix(image=transformed[i])['image'])

  print('saving')
  # save to img_directory
  for i in range(32):
    img_name = img_path.split('/')[-1]
    save_path = img_directory + '/' + img_name.split('.')[0] + '-' + str(i) + '.jpg'
    cv2.imwrite(save_path, transformed[i])


print('saved to ' + img_directory)


