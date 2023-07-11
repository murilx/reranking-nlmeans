import logging
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import random_noise
from skimage.restoration import estimate_sigma
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

import sys
from time import time

from nlmeans.nlmeans_udlf import nlmeans_udlf
from nlmeans.nlmeans import nlmeans
import parameters

hW = parameters.hW
hP = parameters.hP
sig = parameters.sig
tau = parameters.tau
tau = 0.10


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

    # Run filtering nlmeans SAP
    start_time = time()
    im_fil1 = nlmeans(im_nse, hW, hP, tau, sigma_est, shape)
    nlmeans_sap_time = time() - start_time
    
    # Run filtering UDLF nlmeans
    start_time = time()
    im_fil2 = nlmeans_udlf(im_nse, hW, hP, tau, sigma_est, shape)
    nlmeans_uldf_time = time() - start_time

    
    # Show results
    im_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.xlabel('Noise image')
    plt.imshow(im_nse, cmap='gray')
    
    plt.subplot(1, 3, 2)
    plt.xlabel('NLM image')
    plt.imshow(im_fil1, cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.xlabel('LHRR+NLM image')
    plt.imshow(im_fil2, cmap='gray')
    
    plt.savefig('output/' + im_name + '_denoise.png')
    plt.show()
    

    # Evaluate and save info
    level = logging.INFO
    strfmt = "  %(message)s"
    handlers = [logging.FileHandler('output/evaluation.log'),
                logging.StreamHandler()]
    logging.basicConfig(level=level, format=strfmt, handlers=handlers)
    
    logging.info(f"Image: {im_name}")
    logging.info("Non-Local Means SAP:")
    logging.info(f"Time: {nlmeans_sap_time}")
    logging.info(f"PSNR: {psnr(im, im_fil1, data_range=im_fil1.max() - im_fil1.min())}")
    logging.info(f"SSIM: {ssim(im, im_fil1, data_range=im_fil1.max() - im_fil1.min())}")
    logging.info(f"MSE:  {mse(im, im_fil1)}")

    print("\n" + ("-" * 50) + "\n")
    
    # Evaluate UDLF nlmeans
    logging.info("ULDF Non-Local Means:")
    logging.info(f"Time: {nlmeans_uldf_time}")
    logging.info(f"PSNR: {psnr(im, im_fil2, data_range=im_fil2.max() - im_fil2.min())}")
    logging.info(f"SSIM: {ssim(im, im_fil2, data_range=im_fil2.max() - im_fil2.min())}")
    logging.info(f"MSE:  {mse(im, im_fil2)}")
    
if __name__ == '__main__':
    main()
