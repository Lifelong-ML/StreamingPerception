from PIL import Image
import numpy as np

image = Image.open("/Datasets/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0090_796077.jpg")
width, height = image.size
print(width, height)
print(np.array(image).shape)
