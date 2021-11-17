# histogram_equalization
Some histogram equalization methods to enhance image contrast, including AHE and CLAHE.
```diff
+In this fork some optimizations were applied. 
+Especially CLAHE is about 10 times faster comparing to the original implementation.
+Added support of 16-bit png images
```

## Supported methods
  * ImageOps HE
  * HE  
  * AHE
  * CLAHE  
  * Local Region Stretch HE

## Original paper
  * (https://zhuanlan.zhihu.com/p/44918476)

## Code architecture
 * contrast.py  Script for realizing various histogram equalization, ImageContraster class. Also runnable: `python3 contrast.py image_in.jpg image_out.jpg method`
 * main.py, my_main.py Testing scripts

## method
  * The code implements five histogram equalization methods, which are: 1. Histogram equalization using PIL.ImageOps; 2. Histogram equalization HE implemented by yourself; 3. Adaptive histogram equalization AHE; 4. . Contrast-limited adaptive histogram equalization CLAHE; 5. Adaptive local region stretch histogram equalization Local Region Stretch HE. The principle is explained in detail in the link to know.

## Result display
  * Here are some result pictures, each group has six pictures, which are the original picture, ImageOps HE, HE, AHE, CLAHE, Local Region Stretch HE. Results comparison: 
  <div> 
    <table>
     <tr>
      <td><img src = "https://github.com/lxcnju/histogram_equalization/blob/master/pics/car.jpg"></td>
      <td><img src = "https://github.com/lxcnju/histogram_equalization/blob/master/pics/ops_car.jpg"></td>
      <td><img src = "https://github.com/lxcnju/histogram_equalization/blob/master/pics/he_car.jpg"></td>
     </tr>
     <tr>
      <td><img src = "https://github.com/lxcnju/histogram_equalization/blob/master/pics/ahe_car.jpg"></td>
      <td><img src = "https://github.com/lxcnju/histogram_equalization/blob/master/pics/clahe_car.jpg"></td>
      <td><img src = "https://github.com/lxcnju/histogram_equalization/blob/master/pics/lrs_car.jpg"></td>
     </tr>
     <tr>
      <td><img src = "https://github.com/lxcnju/histogram_equalization/blob/master/pics/cap.png"></td>
      <td><img src = "https://github.com/lxcnju/histogram_equalization/blob/master/pics/ops_cap.png"></td>
      <td><img src = "https://github.com/lxcnju/histogram_equalization/blob/master/pics/he_cap.png"></td>
     </tr>
     <tr>
      <td><img src = "https://github.com/lxcnju/histogram_equalization/blob/master/pics/ahe_cap.png"></td>
      <td><img src = "https://github.com/lxcnju/histogram_equalization/blob/master/pics/clahe_cap.png"></td>
      <td><img src = "https://github.com/lxcnju/histogram_equalization/blob/master/pics/lrs_cap.png"></td>
     </tr>
     <tr>
      <td><img src = "https://github.com/lxcnju/histogram_equalization/blob/master/pics/night.png"></td>
      <td><img src = "https://github.com/lxcnju/histogram_equalization/blob/master/pics/ops_night.png"></td>
      <td><img src = "https://github.com/lxcnju/histogram_equalization/blob/master/pics/he_night.png"></td>
     </tr>
     <tr>
      <td><img src = "https://github.com/lxcnju/histogram_equalization/blob/master/pics/ahe_night.png"></td>
      <td><img src = "https://github.com/lxcnju/histogram_equalization/blob/master/pics/clahe_night.png"></td>
      <td><img src = "https://github.com/lxcnju/histogram_equalization/blob/master/pics/lrs_night.png"></td>
     </tr>
    </table>
  </div>
