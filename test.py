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

from parameters import *
from utils import *

# Global variables and constants
OUT_DIR = './output'

im, im_name = get_image(image)
im, im_nse, sigma_est = process_image(im, sig, seed, cut_shapes[im_name])

# Run filtering nlmeans
start_time = time()
im_fil1 = nlmeans(im_nse, hW, hP, tau, sigma_est, shape)
nlmeans_sap_time = time() - start_time

# Run filtering UDLF nlmeans
 udl_method = 'CPRR'
 udl_params = get_udl_params(udl_method)
 start_time = time()
 im_fil2 = nlmeans_udlf(im_nse, hW, hP, tau, sigma_est, shape, udl_method, udl_params, n_w)
 nlmeans_udlf_time = time() - start_time
 rm_tmp_files() # Remove files created by UDLF

# Run filtering NONE nlmeans
udl_method = 'NONE'
udl_params = get_udl_params(udl_method)
start_time = time()
im_fil3 = nlmeans_udlf(im_nse, hW, hP, tau, sigma_est, shape, udl_method, udl_params, n_w)
nlmeans_none_time = time() - start_time
rm_tmp_files() # Remove files created by UDLF (Do not actually needed in this situation)

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
im_noise_psnr, im_noise_ssim, im_noise_mse = image_metrics(im, im_nse)
nlm_noise_psnr, nlm_noise_ssim, nlm_noise_mse = image_metrics(im, im_fil1)
udlf_noise_psnr, udlf_noise_ssim, udlf_noise_mse = image_metrics(im, im_fil2)
none_noise_psnr, none_noise_ssim, none_noise_mse = image_metrics(im, im_fil3)


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
