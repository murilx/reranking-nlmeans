import os
from time import time
import datetime
import random

import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage.color import rgb2gray
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import estimate_sigma
from skimage.util import random_noise

from nlmeans.nlmeans import nlmeans
from nlmeans.nlmeans_udlf import nlmeans_udlf

# Global variables and constants
OUT_DIR = './output'

# Parameters
hW = 10        # Window Size
hP = 3         # Patch Size
tau = 0.15     # Contribution of similarity on noisy-data
sig = 0.1      # Standard deviation of the gaussian noise
shape = 'disk' # Patch shape
n_w = None     # Number of weights to use or use all(None)

# Image to test with
image = 'astronaut'

# UDLF Parameters
udl_method = 'CPRR'
udl_params = {
    'k': 3,
    't': 1
}

def main():
    # List of images that can be selected to test the denoising method
    images = ('astronaut',
              'brick',
              'camera',
              'cat',
              'checkerboard',
              'clock',
              'coffee',
              'coins',
              'grass',
              'gravel',
              'horse')

    # Get the image from skimage.data and do the necessary preprocessing steps
    # In case the image passed is from the file system, just opens it as float
    if image in images:
        im = getattr(data, image)()
        im = im[100:300, 100:300]
        im_name = image
    else:
        im = plt.imread(image).astype('float')
        im_name = os.path.splitext(os.path.basename(image))[0]

    if len(im.shape) == 3:
        im = rgb2gray(im)

        # If the image is one from skimage.data, saves it
        if image == im_name:
            plt.imsave(os.path.join(os.path.dirname(__file__),
                                    'input/skimage', f'{im_name}.png'), im, cmap='gray')

    im_nse = random_noise(im, var = sig**2)

    # estimate the noise standard deviation from the noisy image
    sigma_est = np.mean(estimate_sigma(im_nse, channel_axis=-1))

    # Run filtering nlmeans SAP
    start_time = time()
    im_fil1 = nlmeans(im_nse, hW, hP, tau, sigma_est, shape)
    nlmeans_sap_time = time() - start_time

    # Run filtering UDLF nlmeans
    start_time = time()
    im_fil2 = nlmeans_udlf(im_nse, hW, hP, tau, sigma_est, shape, udl_method, udl_params, n_w)
    nlmeans_udlf_time = time() - start_time

    # Run filtering NONE nlmeans
    start_time = time()
    im_fil3 = nlmeans_udlf(im_nse, hW, hP, tau, sigma_est, shape, 'NONE', None, n_w)
    nlmeans_none_time = time() - start_time

    # Remove the temporary files created
    tmp_files_created = ['input.txt', 'list.txt', 'log.txt', 'output.txt']
    for tmp_file in tmp_files_created:
        try:
            os.remove(tmp_file)
        except FileNotFoundError:
            pass

    # Save the images
    seed = random.randint(100000, 1000000)
    fmt_date = datetime.datetime.now()
    fmt_date = fmt_date.strftime('%d-%m-%Y-%H-%M') # day-month-year-hour-minute

    im_nse_file = os.path.join(OUT_DIR, f'{im_name}_noise_{fmt_date}_{seed}.png')
    plt.imsave(im_nse_file, im_nse, cmap='gray')

    nl_file = os.path.join(OUT_DIR, f'{im_name}_nlmeans_{fmt_date}_{seed}.png')
    plt.imsave(nl_file, im_fil1, cmap='gray')

    udlf_file = os.path.join(OUT_DIR, f'{im_name}_udlf-{udl_method}-{shape}_{fmt_date}_{seed}.png')
    plt.imsave(udlf_file, im_fil2, cmap='gray')

    udlf_file = os.path.join(OUT_DIR, f'{im_name}_udlf-NONE_{fmt_date}_{seed}.png')
    plt.imsave(udlf_file, im_fil3, cmap='gray')

    # Evaluation
    im_noise_psnr   = psnr(im, im_nse, data_range=im_nse.max() - im_nse.min())
    im_noise_ssim   = ssim(im, im_nse, data_range=im_nse.max() - im_nse.min())
    im_noise_mse    = mse(im, im_nse)

    nlm_noise_psnr  = psnr(im, im_fil1, data_range=im_fil1.max() - im_fil1.min())
    nlm_noise_ssim  = ssim(im, im_fil1, data_range=im_fil1.max() - im_fil1.min())
    nlm_noise_mse   = mse(im, im_fil1)

    udlf_noise_psnr = psnr(im, im_fil2, data_range=im_fil2.max() - im_fil2.min())
    udlf_noise_ssim = ssim(im, im_fil2, data_range=im_fil2.max() - im_fil2.min())
    udlf_noise_mse  = mse(im, im_fil2)

    none_noise_psnr = psnr(im, im_fil3, data_range=im_fil3.max() - im_fil3.min())
    none_noise_ssim = ssim(im, im_fil3, data_range=im_fil3.max() - im_fil3.min())
    none_noise_mse  = mse(im, im_fil3)

    # Show information on stdout
    print(f'Seed: {seed}')
    print(f'Method: {udl_method}')
    print()

    print('Noised Image:')
    print(f'PSNR: {im_noise_psnr}')
    print(f'SSIM: {im_noise_ssim}')
    print(f'MSE : {im_noise_mse}')
    print()

    print('Non-Local Means:')
    print(f'Time: {nlmeans_sap_time}')
    print(f'PSNR: {nlm_noise_psnr}')
    print(f'SSIM: {nlm_noise_ssim}')
    print(f'MSE : {nlm_noise_mse}')
    print()

    print('NONE Non-Local Means:')
    print(f'Time: {nlmeans_none_time}')
    print(f'PSNR: {none_noise_psnr}')
    print(f'SSIM: {none_noise_ssim}')
    print(f'MSE : {none_noise_mse}')
    print()

    print('UDLF Non-Local Means:')
    print(f'Time: {nlmeans_udlf_time}')
    print(f'PSNR: {udlf_noise_psnr}')
    print(f'SSIM: {udlf_noise_ssim}')
    print(f'MSE : {udlf_noise_mse}')
    print()


if __name__ == '__main__':
    main()
