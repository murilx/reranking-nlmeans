import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import estimate_sigma
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import cv2 as cv

import sys
from time import time

import parameters

hW = parameters.hW
hP = parameters.hP
sig = parameters.sig
sig = 50
mean = 0.0
h = 45


def main():
    # Load the image and add noise to it
    im = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)
    noise = np.copy(im)
    cv.randn(noise, mean, sig)
    im_nse = cv.add(im, noise)
    
    # im = plt.imread(sys.argv[1]).astype('float')
    # im_nse = img_as_ubyte(random_noise(im, var=sig ** 2))

    # estimate the noise standard deviation from the noisy image
    sigma_est = np.mean(estimate_sigma(im_nse, channel_axis=-1))
    print('estimated noise standard deviation =', sigma_est)

    # Run filtering
    start_time = time()
    im_fil = cv.fastNlMeansDenoising(im_nse, None, h, hP, 2*hW+1)
    print("Time spent: ", time() - start_time)

    # Show results
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im_nse, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(im_fil, cmap='gray')
    plt.show()

    # Evaluate
    print("PSNR: ", psnr(im, im_fil, data_range=im_fil.max() - im_fil.min()))
    print("SSIM: ", ssim(im, im_fil, data_range=im_fil.max() - im_fil.min()))
    print("MSE:  ", mse(im, im_fil))
    
if __name__ == '__main__':
    main()
