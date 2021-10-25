#-*- coding : utf-8 -*-

from imageio import imread, imwrite
import numpy as np
from contrast import ImageContraster

basename = "car"

# read image
img_arr: np.ndarray = imread(basename+'.png', 'PNG-FI')
print("Input pixel type: ", img_arr.dtype.name)
print(img_arr.shape)
level = 256 if img_arr.dtype.name == "uint8" else 65536
# contraster
icter = ImageContraster()

def he():
    print("Compute HE")
    he_eq_img = icter.enhance_contrast(img_arr, level, method = "HE")
    print("saving HE..")
    imwrite(basename+"_HE.png", he_eq_img, 'PNG-FI')

def ahe():
    print("Compute AHE")
    ahe_eq_img = icter.enhance_contrast(img_arr, level, method = "AHE", window_size = 32, affect_size = 16)
    print("saving AHE..")
    imwrite(basename+"_AHE.png", ahe_eq_img, 'PNG-FI')

def clahe():
    print("Compute CLAHE")
    clahe_eq_img = icter.enhance_contrast(img_arr, level, method = "CLAHE", blocks = 8, threshold = 10.0)
    print("saving CLAHE..")
    imwrite(basename+"_CLAHE.png", clahe_eq_img, 'PNG-FI')

def lrs():
    print("Compute Local Region Stretch")
    lrs_eq_img = icter.enhance_contrast(img_arr, level, method = "Bright")
    print("saving Local Region Stretch..")
    imwrite(basename+"_LRS.png", lrs_eq_img, 'PNG-FI')

clahe()

# procs = [Process(target=he), Process(target=ahe), Process(target=clahe), Process(target=lrs)]
#
# for proc in procs:
#     proc.start()
#
# with Pool(processes=4) as pool:
#     [pool.apply_async(proc.join) for proc in procs]
#     pool.close()
#     pool.join()
