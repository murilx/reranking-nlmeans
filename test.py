import argparse
import logging
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage.color import rgb2gray
from skimage.util import random_noise
from skimage.restoration import estimate_sigma
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

from time import time

from nlmeans.nlmeans_udlf import nlmeans_udlf
from nlmeans.nlmeans import nlmeans
import parameters

hW = parameters.hW
hP = parameters.hP
sig = parameters.sig
tau = parameters.tau


def main():
    # List of images that can be selected to test the denoising methods
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
    
    parser = argparse.ArgumentParser(
        description='Test and compare denoising methods.',
        prog='test.py')

    parser.add_argument('-i', '--image', default=['astronaut'], nargs=1,
                        help='image to be used in the test.')
    parser.add_argument('-p', '--patch', choices=['disk', 'square'],
                        default=['square'], nargs=1,
                        help='patch shape to used in the test')

    args = parser.parse_args()

    # Get the image from skimage.data and do the necessary preprocessing steps
    # In case the image passed is from the file system, just opens it as float
    if args.image[0] in images:
        im = getattr(data, args.image[0])()
        im = im[100:300, 100:300]
        im_name = args.image[0]
    else:
        im = plt.imread(sys.argv[1]).astype('float')
        im_name = os.path.splitext(os.path.basename(sys.argv[1]))[0]

    # If image is rgb converts it to grayscale
    if len(im.shape) == 3:
        im = rgb2gray(im)
    
    im_nse = random_noise(im, var=sig ** 2)

    # estimate the noise standard deviation from the noisy image
    sigma_est = np.mean(estimate_sigma(im_nse, channel_axis=-1))

    shape = args.patch[0]

    # Run filtering nlmeans SAP
    start_time = time()
    im_fil1 = nlmeans(im_nse, hW, hP, tau, sigma_est, shape)
    nlmeans_sap_time = time() - start_time
    
    # Run filtering UDLF nlmeans
    start_time = time()
    im_fil2 = nlmeans_udlf(im_nse, hW, hP, tau, sigma_est, shape)
    nlmeans_uldf_time = time() - start_time
    
    # Remove the temporary files created
    tmp_files_created = ['input.txt', 'list.txt', 'log.txt', 'output.txt']
    for tmp_file in tmp_files_created:
        try:
            os.remove(tmp_file)
        except FileNotFoundError:
            pass

    # Show results
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
    strfmt = '  %(message)s'
    handlers = [logging.FileHandler('output/evaluation.log'),
                logging.StreamHandler()]
    logging.basicConfig(level=level, format=strfmt, handlers=handlers)
    
    logging.info(f'Image: {im_name}')
    logging.info(f'estimated noise standard deviation: {sigma_est}')

    logging.info('Noised Image:')
    logging.info(f'PSNR: {psnr(im, im_nse, data_range=im_nse.max() - im_nse.min())}')
    logging.info(f'SSIM: {ssim(im, im_nse, data_range=im_nse.max() - im_nse.min())}')
    logging.info(f'MSE:  {mse(im, im_nse)}')
    print('\n' + ('-' * 50) + '\n')

    logging.info('Non-Local Means SAP:')
    logging.info(f'Time: {nlmeans_sap_time}')
    logging.info(f'PSNR: {psnr(im, im_fil1, data_range=im_fil1.max() - im_fil1.min())}')
    logging.info(f'SSIM: {ssim(im, im_fil1, data_range=im_fil1.max() - im_fil1.min())}')
    logging.info(f'MSE:  {mse(im, im_fil1)}')

    print('\n' + ('-' * 50) + '\n')
    
    logging.info('ULDF Non-Local Means:')
    logging.info(f'Time: {nlmeans_uldf_time}')
    logging.info(f'PSNR: {psnr(im, im_fil2, data_range=im_fil2.max() - im_fil2.min())}')
    logging.info(f'SSIM: {ssim(im, im_fil2, data_range=im_fil2.max() - im_fil2.min())}')
    logging.info(f'MSE:  {mse(im, im_fil2)}')
    

if __name__ == '__main__':
    main()
