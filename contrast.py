# -*- coding:utf-8 -*-

from PIL import Image, ImageOps
import numpy as np
from matplotlib import pyplot as plt


class ImageContraster():
    def __init__(self):
        pass

    def enhance_contrast(self, img_arr: np.array_repr, level, method: str=None, window_size=None, affect_size=None,
                         blocks=None,
                         threshold=None):
        ### equalize the histogram
        ### @params img : Image type
        ### @params method : Histogram Equalization Method
        ### @params level : color or gray scale
        ### @params window_size : in AHE, the window to calculate Histogram CDF size
        ### @params affect_size : in AHE, the affected pixels size
        ### @params blocks : in CLAHE, split how many times in row and col
        ### @params threshold : in CLAHE, if threshold times higher then the mean value, clip 
        ### @return img_res : equalized result

        # choose algorithms
        if method in ["HE", "FHE", "he", "fhe"]:
            he_func = self.histogram_equalization  # HE
        elif method in ["AHE", "ahe"]:
            he_func = self.adaptive_histequal  # AHE
        elif method in ["CLAHE", "clahe"]:
            he_func = self.contrast_limited_ahe  # CLAHE
        elif method in ["standard", "STANDARD", "Standard"]:
            he_func = self.standard_histogram_equalization  # ImageOps HE
        elif method in ["Bright", "bright", "bright_level"]:
            he_func = self.bright_wise_histequal  # Local Region Stretch

        # process gray and color images
        if len(img_arr.shape) == 2:
            channel_num = 1
        elif len(img_arr.shape) == 3:
            channel_num = img_arr.shape[2]

        if channel_num == 1:
            # gray image
            arr = he_func(img_arr, level, window_size=window_size, affect_size=affect_size, blocks=blocks,
                          threshold=threshold)
            img_res = arr
        elif channel_num == 3 or channel_num == 4:
            # RGB image or RGBA image(such as png)
            rgb_arr = [None] * 3

            orig_type = img_arr.dtype.name
            m, n, _ = img_arr.shape

            float_arr = img_arr.astype(np.float)
            # y_arr = np.sum((float_arr * np.array([0.299, 0.586, 0.114]).reshape((1,1,3)) ), axis=2) # Y part of YUV
            # y_arr = np.sum(float_arr, axis=2) / 3.0 # Y part of YUV
            y_arr = np.max(float_arr, axis=2)  # Y part of YUV
            y_level = max(level, int(np.ceil(y_arr.max())) + 1)

            equalized_y_arr = he_func(y_arr.round().astype(orig_type), y_level, window_size=window_size,
                                      affect_size=affect_size,
                                      blocks=blocks, threshold=threshold)

            y_arr[y_arr < 0.5] = 0.5

            fullcfs = equalized_y_arr.reshape((m, n, 1)) / y_arr.reshape((m, n, 1))

            fullcfs[fullcfs<1.0] = np.power(fullcfs[fullcfs<1.0], 0.05)

            img_res_float = (float_arr * fullcfs)
            # img_res_float = (float_arr + equalized_y_arr.reshape((m, n, 1)) - y_arr.reshape((m, n, 1)))

            min_val = img_res_float.min()

            if min_val < 0:
                img_res_float += -min_val

            cf = float(level - 1) / img_res_float.max()

            img_res_float *= cf

            img_res = img_res_float.round().astype(orig_type)

            ## process dividely
            # for k in range(3):
            #    rgb_arr[k] = he_func(img_arr[:, :, k], level, window_size=window_size, affect_size=affect_size,
            #                         blocks=blocks, threshold=threshold)
            # img_res = np.array(rgb_arr).transpose((1, 2, 0))

        return img_res

    def histogram_equalization(self, img_arr, level, **args):
        ### equalize the distribution of histogram to enhance contrast
        ### @params img_arr : numpy.array uint8 type, 2-dim
        ### @params level : the level of gray scale
        ### @return arr : the equalized image array

        orig_type = img_arr.dtype.name

        # calculate hists
        hists = self.calc_histogram_(img_arr, level)

        # equalization
        (m, n) = img_arr.shape
        hists_cdf = self.calc_histogram_cdf_(hists, m, n, level, orig_type)  # calculate CDF

        arr = hists_cdf[img_arr]  # mapping

        return arr

    def adaptive_histequal(self, img_arr, level, window_size=32, affect_size=16, **args):
        ### using AHE to enhance contrast
        ### @params img_arr : numpy.array uint8 type, 2-dim
        ### @params level : the level of gray scale
        ### @params window_size : calculate the histogram mapping function in a window of (window_size, window_size)
        ### @params affect_size : the real affected pixels by a mapping function
        ### @return arr : the equalized image array
        arr = img_arr.copy()

        # calculate how many blocks needed in row-axis and col-axis
        (m, n) = img_arr.shape
        if (m - window_size) % affect_size == 0:
            rows = int((m - window_size) / affect_size + 1)
        else:
            rows = int((m - window_size) / affect_size + 2)
        if (n - window_size) % affect_size == 0:
            cols = int((n - window_size) / affect_size + 1)
        else:
            cols = int((n - window_size) / affect_size + 2)

        # equalize histogram of every block image
        for i in range(rows):
            for j in range(cols):
                # offset
                off = int((window_size - affect_size) / 2)

                # affect region border
                asi, aei = i * affect_size + off, (i + 1) * affect_size + off
                asj, aej = j * affect_size + off, (j + 1) * affect_size + off

                # window region border
                wsi, wei = i * affect_size, i * affect_size + window_size
                wsj, wej = j * affect_size, j * affect_size + window_size

                # equalize the window region
                window_arr = img_arr[wsi: wei, wsj: wej]
                block_arr = self.histogram_equalization(window_arr, level)

                # border case
                if i == 0:
                    arr[wsi: asi, wsj: wej] = block_arr[0: asi - wsi, :]
                elif i >= rows - 1:
                    arr[aei: wei, wsj: wej] = block_arr[aei - wsi: wei - wsi, :]
                if j == 0:
                    arr[wsi: wei, wsj: asj] = block_arr[:, 0: asj - wsj]
                elif j >= cols - 1:
                    arr[wsi: wei, aej: wej] = block_arr[:, aej - wsj: wej - wsj]
                arr[asi: aei, asj: aej] = block_arr[asi - wsi: aei - wsi, asj - wsj: aej - wsj]

        return arr

    def contrast_limited_ahe(self, img_arr, level, blocks, threshold, **args):
        ### equalize the distribution of histogram to enhance contrast, using CLAHE
        ### @params img_arr : numpy.array uint8 type, 2-dim
        ### @params level : the level of gray scale
        ### @params window_size : the window used to calculate CDF mapping function
        ### @params threshold : clip histogram by exceeding the threshold times of the mean value
        ### @return arr : the equalized image array
        (m, n) = img_arr.shape
        block_m = 2 * int(np.ceil(m / (2 * blocks)))
        block_n = 2 * int(np.ceil(n / (2 * blocks)))
        orig_type = img_arr.dtype.name

        # split small regions and calculate the CDF for each, save to a 2-dim list
        maps = []
        for i in range(blocks):
            row_maps = []
            for j in range(blocks):
                # block border
                si, ei = i * block_m, (i + 1) * block_m
                sj, ej = j * block_n, (j + 1) * block_n

                # block image array
                block_img_arr = img_arr[si: ei, sj: ej]

                # calculate histogram and cdf
                hists = self.calc_histogram_(block_img_arr, level)
                clip_hists = self.clip_histogram_(hists, threshold=threshold)  # clip histogram
                hists_cdf = self.calc_histogram_cdf_(clip_hists, block_m * block_n, level, orig_type)

                # hmax = hists_cdf.max()
                # max_idx = len(hists_cdf[hists_cdf<hmax])+1
                # self.draw_histograms_([hists[:max_idx], clip_hists[:max_idx], hists_cdf[:max_idx]])
                # exit(0)
                # save
                row_maps.append(hists_cdf)
            maps.append(row_maps)
        maps: np.ndarray = np.array(maps)

        # interpolate every pixel using four nearest mapping functions
        # pay attention to border case
        arr = img_arr.copy()

        block_m_step = round(block_m / 2)
        block_n_step = round(block_n / 2)

        for m_start in range(0, m, block_m_step):
            for n_start in range(0, n, block_n_step):
                ri = (m_start, min(m_start + block_m_step, m))
                rj = (n_start, min(n_start + block_n_step, n))
                range_i = range(m_start, min(m_start + block_m_step, m))
                range_j = range(n_start, min(n_start + block_n_step, n))
                arr_i = np.array(range_i)
                arr_j = np.array(range_j)

                arr_r: np.ndarray = np.floor((arr_i.astype(np.float32) - block_m_step) / block_m).astype(np.int)
                arr_c: np.ndarray = np.floor((arr_j.astype(np.float32) - block_n_step) / block_n).astype(np.int)

                arr_r_u = np.unique(arr_r)
                arr_c_u = np.unique(arr_c)

                assert arr_r_u.shape[0] == 1
                assert arr_c_u.shape[0] == 1

                rl: int = arr_r_u[0]
                cl: int = arr_c_u[0]

                arr_x1: np.ndarray = (
                            (arr_i.astype(np.float32) - (arr_r.astype(np.float32) + 0.5) * block_m) / block_m).astype(
                    np.float32)
                arr_y1: np.ndarray = (
                            (arr_j.astype(np.float32) - (arr_c.astype(np.float32) + 0.5) * block_n) / block_n).astype(
                    np.float32)

                arr_x1_sub = (1.0 - arr_x1)
                arr_y1_sub = (1.0 - arr_y1)

                # Case r < 0 and c < 0:
                if rl < 0 and cl < 0:
                    img_arr_idx: np.ndarray = img_arr[ri[0]:ri[1],rj[0]:rj[1]]
                    arr[ri[0]:ri[1],rj[0]:rj[1]] = maps[rl + 1][cl + 1][img_arr_idx]
                # Case r < 0 and c >= blocks - 1:
                elif rl < 0 and cl >= blocks - 1:
                    img_arr_idx: np.ndarray = img_arr[ri[0]:ri[1],rj[0]:rj[1]]
                    arr[ri[0]:ri[1],rj[0]:rj[1]] = maps[rl + 1][cl][img_arr_idx]
                # Case r >= blocks - 1 and c < 0:
                elif rl >= blocks - 1 and cl < 0:
                    img_arr_idx: np.ndarray = img_arr[ri[0]:ri[1],rj[0]:rj[1]]
                    arr[ri[0]:ri[1],rj[0]:rj[1]] = maps[rl][cl + 1][img_arr_idx]
                # Case r >= blocks - 1 and c >= blocks - 1:
                elif rl >= blocks - 1 and cl >= blocks - 1:
                    img_arr_idx: np.ndarray = img_arr[ri[0]:ri[1],rj[0]:rj[1]]
                    arr[ri[0]:ri[1],rj[0]:rj[1]] = maps[rl][cl][img_arr_idx]
                # Case r < 0 or r >= blocks - 1:
                elif rl < 0 or rl >= blocks - 1:
                    img_arr_idx: np.ndarray = img_arr[ri[0]:ri[1],rj[0]:rj[1]]
                    rc = min(max(rl, 0), blocks - 1)
                    mapped_left: np.ndarray = arr_y1_sub * maps[rc][cl][img_arr_idx]
                    mapped_right: np.ndarray = arr_y1 * maps[rc][cl + 1][img_arr_idx]
                    mapped_mult_sum: np.ndarray = mapped_left + mapped_right  # + mapped_v
                    arr[ri[0]:ri[1],rj[0]:rj[1]] = mapped_mult_sum
                # Case c < 0 or c >= blocks - 1:
                elif cl < 0 or cl >= blocks - 1:
                    img_arr_idx: np.ndarray = img_arr[ri[0]:ri[1],rj[0]:rj[1]]
                    cc = min(max(cl, 0), blocks - 1)
                    mapped_up = maps[rl][cc][img_arr_idx]
                    mapped_bottom = maps[rl + 1][cc][img_arr_idx]
                    mapped_mult_sum: np.ndarray = mapped_up * arr_x1_sub[:,np.newaxis] + mapped_bottom * arr_x1[:,np.newaxis]
                    arr[ri[0]:ri[1],rj[0]:rj[1]] =  mapped_mult_sum
                # Inner case
                else:
                    img_arr_idx: np.ndarray = img_arr[ri[0]:ri[1],rj[0]:rj[1]]
                    mapped_lu = maps[rl][cl][img_arr_idx]
                    mapped_lb = maps[rl + 1][cl][img_arr_idx]
                    mapped_ru = maps[rl][cl + 1][img_arr_idx]
                    mapped_rb = maps[rl + 1][cl + 1][img_arr_idx]
                    mapped_mult_sum: np.ndarray = arr_y1_sub * (arr_x1_sub[:,np.newaxis] * mapped_lu + arr_x1[:,np.newaxis] * mapped_lb) \
                                                  + arr_y1 * (arr_x1_sub[:,np.newaxis] * mapped_ru + arr_x1[:,np.newaxis] * mapped_rb)
                    arr[ri[0]:ri[1],rj[0]:rj[1]] = mapped_mult_sum
        return arr.astype(orig_type)

    def bright_wise_histequal(self, img_arr, level, **args):
        ### split the image to three level accoding brightness, equalize histogram dividely
        ### @params img_arr : numpy.array uint8 type, 2-dim
        ### @params level : gray scale
        ### @return arr : the equalized image array

        orig_type = img_arr.dtype.name

        def special_histogram(img_arr, min_v, max_v):
            ### calculate a special histogram with max, min value
            ### @params img_arr : 1-dim numpy.array
            ### @params min_v : min gray scale
            ### @params max_v : max gray scale
            ### @return hists : list type, length = max_v - min_v + 1
            hists = [0 for _ in range(max_v - min_v + 1)]
            for v in img_arr:
                hists[v - min_v] += 1
            return hists

        def special_histogram_cdf(hists, min_v, max_v):
            ### calculate a special histogram cdf with max, min value
            ### @params hists : list type
            ### @params min_v : min gray scale
            ### @params max_v : max gray scale
            ### @return hists_cdf : numpy.array
            hists_cumsum = np.cumsum(np.array(hists))
            hists_cdf = (max_v - min_v) / hists_cumsum[-1] * hists_cumsum + min_v
            hists_cdf = hists_cdf.astype(orig_type)
            return hists_cdf

        def pseudo_variance(arr):
            ### caluculate a type of variance
            ### @params arr : 1-dim numpy.array
            arr_abs = np.abs(arr - np.mean(arr))
            return np.mean(arr_abs)

        # search two grayscale level, which can split the image into three parts having approximately same number of pixels
        (m, n) = img_arr.shape
        hists = self.calc_histogram_(img_arr, level)
        hists_arr = np.cumsum(np.array(hists))
        hists_ratio = hists_arr / hists_arr[-1]

        scale1 = None
        scale2 = None
        for i in range(len(hists_ratio)):
            if hists_ratio[i] >= 0.333 and scale1 == None:
                scale1 = i
            if hists_ratio[i] >= 0.667 and scale2 == None:
                scale2 = i
                break

        # split images
        dark_index = (img_arr <= scale1)
        mid_index = (img_arr > scale1) & (img_arr <= scale2)
        bright_index = (img_arr > scale2)

        # variance
        dark_variance = pseudo_variance(img_arr[dark_index])
        mid_variance = pseudo_variance(img_arr[mid_index])
        bright_variance = pseudo_variance(img_arr[bright_index])

        # build three level images
        dark_img_arr = np.zeros_like(img_arr)
        mid_img_arr = np.zeros_like(img_arr)
        bright_img_arr = np.zeros_like(img_arr)

        # histogram equalization individually
        dark_hists = special_histogram(img_arr[dark_index], 0, scale1)
        dark_cdf = special_histogram_cdf(dark_hists, 0, scale1)

        mid_hists = special_histogram(img_arr[mid_index], scale1, scale2)
        mid_cdf = special_histogram_cdf(mid_hists, scale1, scale2)

        bright_hists = special_histogram(img_arr[bright_index], scale2, level - 1)
        bright_cdf = special_histogram_cdf(bright_hists, scale2, level - 1)

        def plot_hists(arr):
            hists = [0 for i in range(256)]
            for a in arr:
                hists[a] += 1
            self.draw_histogram_(hists)

        # mapping
        dark_img_arr[dark_index] = dark_cdf[img_arr[dark_index]]
        mid_img_arr[mid_index] = mid_cdf[img_arr[mid_index] - scale1]
        bright_img_arr[bright_index] = bright_cdf[img_arr[bright_index] - scale2]

        # weighted sum
        # fractor = dark_variance + mid_variance + bright_variance
        # arr = (dark_variance * dark_img_arr + mid_variance * mid_img_arr + bright_variance * bright_img_arr)/fractor
        arr = dark_img_arr + mid_img_arr + bright_img_arr
        arr = arr.astype(orig_type)
        return arr

    def standard_histogram_equalization(self, img_arr, level, **args):
        ### equalize the distribution of histogram to enhance contrast, using PIL.ImageOps
        ### @params img_arr : numpy.array uint8 type, 2-dim
        ### @params level : the level of gray scale
        ### @return arr : the equalized image array

        # ImageOps.equalize
        img = Image.fromarray(img_arr)
        img_res = ImageOps.equalize(img)
        arr = np.array(img_res)
        return arr

    def calc_histogram_(self, gray_arr, level):
        ### calculate the histogram of a gray scale image
        ### @params gray_arr : numpy.array uint8 type, 2-dim
        ### @params level : the level of gray scale
        ### @return hists : list type
        hists = np.zeros((level), dtype=int)
        uniq_idxs, counts = np.unique(gray_arr, return_counts=True)
        hists[uniq_idxs] += counts
        return hists

    def calc_histogram_cdf_(self, hists, block_mn, level, orig_type):
        ### calculate the CDF of the hists
        ### @params hists : list type
        ### @params block_m : the histogram block's height
        ### @params block_n : the histogram block's width
        ### @params level : the level of gray scale
        ### @return hists_cdf : numpy.array type
        first_nz, last_nz = np.argwhere(hists>3)[[0, -1]]
        hists_cumsum = np.cumsum(np.array(hists)) + hists[first_nz]

        # Limit contrast range to near original one
        max_level = (level - 1 + last_nz) / 2
        min_level = first_nz / 2

        const_a = max_level / max(hists_cumsum.max(), 1)
        hists_cdf = (const_a * hists_cumsum)

        cf = (float(max_level) - min_level) / max_level

        hists_cdf *= cf
        hists_cdf += min_level

        return hists_cdf.round().astype(orig_type)

    def clip_histogram_(self, hists, threshold):
        ### clip the peak of histogram, and separate it to all levels uniformly
        ### @params hists : list type
        ### @params threshold : the top ratio of hists over mean value
        ### @return clip_hists : list type
        all_sum = sum(hists)
        threshold_value = all_sum / len(hists) * threshold
        total_extra = (hists[hists >= threshold_value] - threshold_value).sum()
        mean_extra = total_extra / len(hists)

        clip_hists = np.zeros((len(hists)), dtype=int)
        clip_hists[hists >= threshold_value] = int(threshold_value + mean_extra)
        clip_hists[hists < threshold_value] = (hists[hists < threshold_value] + mean_extra).astype(int)

        return clip_hists

    def draw_histogram_(self, hist):
        ### draw a bar picture of the given histogram
        plt.figure()
        plt.bar(range(len(hist)), hist)
        plt.show()

    def draw_histograms_(self, hists):
        ### draw a bar picture of the given histogram
        plt.figure()
        for hist in hists:
            plt.plot(range(len(hist)), hist)
        plt.show()

    def plot_images(self, img1, img2):
        ### draw two images
        ### @params img1 : Image type
        ### @params img2 : Image type
        plt.figure()
        plt.subplot(121)
        plt.imshow(img1)
        plt.subplot(122)
        plt.imshow(img2)
        plt.show()
