#!/usr/bin/python3
from PIL import Image, ImageDraw
import numpy as np
import random


save_path = "/scratch/ssolit/stream_data/noise_data/"
desired_images = 1000

generate_noise = True
rect_num = 0
rect_colors = ["yellow"]
ellipse_num = 0
ellipse_colors = ["red"]


rgb_dict = {
  "red" : (255, 0, 0),
  "orange" : (255, 128, 0),
  "yellow" : (255, 255, 0),
  "green" : (0, 255, 0),
  "blue" : (0, 0, 255),
  "purple" : (150, 50, 255),
  "white" : (255, 255, 255)
}

'''
print("testing")
noise_sequence = []
noise_sequence.append((1, 2, 3))
noise_sequence.append((2, 5, 6))
print(noise_sequence)
print("end testing")
newdata = list(range(0, 256, 4)) * 104
print(newdata)
'''

def get_bounding_coords():
    x0 = random.randint(0, 255)
    y0 = random.randint(0, 127)
    x1 = x0 + random.randint(0, 255 - x0)
    y1 = y0 + random.randint(0, 127 - y0)
    return [x0, y0, x1, y1]


for i in range(desired_images):
    im= Image.new("RGB", (256, 128), "#000000")
    name = "image_" + '{:03}'.format(i)
    if (generate_noise):
        noise_sequence = []
        for x in range(0, 255):
            for y in range (0, 127):
                if (random.randint(0, 1)):
                    im.putpixel((x,y), (255, 255, 255))

    # generate rectangles
    for j in range(rect_num):
        coords = get_bounding_coords()
        color = rgb_dict[random.choice(rect_colors)]
        draw = ImageDraw.Draw(im)
        draw.rectangle(coords, fill=color)

    # generate ellipses
    for j in range(ellipse_num):
        coords = get_bounding_coords()
        color = rgb_dict[random.choice(ellipse_colors)]
        draw = ImageDraw.Draw(im)
        draw.ellipse(coords, fill=color)



    im.save(save_path + name + ".jpg", 'JPEG')

print("finished")
