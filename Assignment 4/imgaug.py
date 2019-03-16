# Image augmentation

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np

IMAGE = './linus.jpg' # Path of input image
OUTPUT = './Outputs' # Output directory
PREFIX = '' # Prefix to output filename

image = load_img(IMAGE)
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

datagen = ImageDataGenerator(
    rotation_range=30, 
    width_shift_range=0.1,
    height_shift_range=0.1, 
    shear_range=0.2, 
    zoom_range=0.2,
    horizontal_flip=True, 
    fill_mode="nearest")

total = 0

gen = datagen.flow(image, save_to_dir=OUTPUT, save_format='jpg')

for img in gen:
	total += 1

	if total == 10:
		break