import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

import sys
from time import time

from nlmeans_sar import nlmeans_sar
from nlmeans_sar_it import nlmeans_sar_it


def main():
    hW = 10    # Window size
    hP = 3     # Patch size
    tau = 0.15 # Contribution of similarity on noisy-data
    T = 0.15   # Contribution of similarity on pre-estimated data
    L = 3      # Number of look
    N = 4      # Number of iteration

    # Load image and synthetize Nakagami-Rayleigh noise of parameter L
    im = plt.imread(sys.argv[1]).astype('float')
    im_nse = np.sqrt(stats.gamma.rvs(L, scale=im**2/L))

    # Run filtering
    start_time = time()
    im_fil = nlmeans_sar(im_nse, hW, hP, tau)
    for i in range(1, N):
        im_fil = nlmeans_sar_it(im_nse, im_fil, hW, hP, tau, T)
    print("Time spent: ", time() - start_time)

    # Show results
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im_nse, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(im_fil, cmap='gray')
    plt.show()

    # Evaluate
    print("PSNR: ", psnr(im, im_fil))
    print("SSIM: ", ssim(im, im_fil))
    print("MSE: ", mse(im, im_fil))
    
if __name__ == '__main__':
    main()
