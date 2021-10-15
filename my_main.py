#-*- coding : utf-8 -*-

from imageio import imread, imwrite
from multiprocessing import Pool

from contrast import ImageContraster

basename = "/tmp/glsl_debayer_rgb_HDR_br"

# read image
img_arr = imread(basename+'.png', 'PNG-FI')
print("Input pixel type: ", img_arr.dtype.name)
print(img_arr.shape)
level = 65536
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




with Pool(processes=4) as pool:
    pool.apply_async(he)
    pool.apply_async(ahe)
    pool.apply_async(clahe)
    pool.apply_async(lrs)
    pool.close()
    pool.join()
