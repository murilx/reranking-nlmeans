import matplotlib.pyplot as plt
import numpy as np
from skimage.util import random_noise
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

import sys
from time import time


def main():
    hW = 10    # Window size
    hP = 3     # Patch size

    # Load image and synthetize Nakagami-Rayleigh noise of parameter L
    im = plt.imread(sys.argv[1]).astype('float')
    im_nse = random_noise(im, var=0.08**2)

    # estimate the noise standard deviation from the noisy image
    sigma_est = np.mean(estimate_sigma(im_nse, channel_axis=-1))
    print('estimated noise standard deviation =', sigma_est)

    # nlmeans parameters
    patch_kw = dict(patch_size = hP,
                    patch_distance = hW//2,
                    channel_axis=None) # grayscale image
    
    # Run filtering
    start_time = time()
    im_fil = denoise_nl_means(im_nse, h=0.8 * sigma_est, sigma=sigma_est,
                              fast_mode=False, **patch_kw)
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
