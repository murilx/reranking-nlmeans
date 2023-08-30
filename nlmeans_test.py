import matplotlib.pyplot as plt
import numpy as np
from skimage.util import random_noise
from skimage.restoration import estimate_sigma
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

import sys
from time import time

from nlmeans.nlmeans import nlmeans
import parameters

hW = parameters.hW
hP = parameters.hP
sig = parameters.sig
tau = parameters.tau


def main():
    # Load the image and add noise to it
    im = plt.imread(sys.argv[1]).astype('float')
    im_nse = random_noise(im, var=sig ** 2)

    # estimate the noise standard deviation from the noisy image
    sigma_est = np.mean(estimate_sigma(im_nse, channel_axis=-1))
    print('estimated noise standard deviation =', sigma_est)

    # Determine the patch shape
    shape = 'square'
    if(len(sys.argv) == 3):
        shape = sys.argv[2]

    # Run filtering
    start_time = time()
    im_fil = nlmeans(im_nse, hW, hP, tau, sigma_est, shape)
    # im_fil = nlmeans(im_nse, hW, hP, tau, sig, shape)
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
