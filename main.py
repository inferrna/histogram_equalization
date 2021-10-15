#-*- coding : utf-8 -*-

from PIL import Image, ImageOps
import numpy as np

from contrast import ImageContraster

# read image
img = Image.open("car.jpg")
print(np.array(img).shape)

np_arr = np.array(img)
level = 65536 if np_arr.dtype.name == 'uint16' else 256

# contraster
icter = ImageContraster()

# HE
he_eq_img = icter.enhance_contrast(np_arr, level, method = "HE")
icter.plot_images(img, he_eq_img)

# AHE
ahe_eq_img = icter.enhance_contrast(np_arr, level, method = "AHE", window_size = 32, affect_size = 16)
icter.plot_images(img, ahe_eq_img)

# CLAHE
clahe_eq_img = icter.enhance_contrast(np_arr, level, method = "CLAHE", blocks = 8, threshold = 10.0)
icter.plot_images(img, clahe_eq_img)

# Local Region Stretch
lrs_eq_img = icter.enhance_contrast(np_arr, level, method = "Bright")
icter.plot_images(img, lrs_eq_img)


