import os
import matplotlib.pyplot as plt
import numpy as np

from skimage import data
from skimage.color import rgb2gray
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import estimate_sigma
from skimage.util import random_noise


# List of images that can be selected to test the denoising method
images = (
    'astronaut',
    'brick',
    'camera',
    'cat',
    'checkerboard',
    'clock',
    'coffee',
    'coins',
    'grass',
    'gravel',
    'horse'
)

# Dictionary containing the "optimal" cut shape for each image
cut_shapes = {
    'astronaut': (100, 300),
    'brick': (100, 300),
    'camera': (80, 280),
    'cat': (100, 300) ,
    'checkerboard': None,
    'clock': (60, 260, 110, 310),
    'coffee': (150, 350),
    'coins': (100, 300),
    'grass': None,
    'gravel': None,
    'horse': (50, 250)
}


def get_image(image):
    # Get the image from skimage.data and do the necessary preprocessing steps
    # In case the image passed is from the file system, just opens it as float
    if image in images:
        im = getattr(data, image)()
        im_name = image
        if not np.issubdtype(im.dtype, np.floating):
            im = im.astype(float)
            im = (im - np.min(im)) / (np.max(im) - np.min(im))
    else:
        im = plt.imread(image).astype('float')
        im_name = os.path.splitext(os.path.basename(image))[0]

    return im, im_name


def process_image(im, sig, seed = None, cut_shape = None):
    """Process the image by cutting, changing it to grayscale and adding noise.
    """
    # Cuts the image to be a square image with 200x200 pixels max
    if cut_shape == None:
        if im.shape[0] > 200:
            if im.shape[1] < 200:
                im = im[0:im.shape[1], :]
            else:
                im = im[0:200, :]
    
        if im.shape[1] > 200:
            if im.shape[0] < 200:
                im = im[:, 0:im.shape[0]]
            else:
                im = im[:, 0:200]
    elif len(cut_shape) == 2:
        im = im[cut_shape[0]:cut_shape[1], cut_shape[0]:cut_shape[1]]
    elif len(cut_shape) == 4:
        im = im[cut_shape[0]:cut_shape[1], cut_shape[2]:cut_shape[3]]
    else:
        print('[err] Invalid cut_shape parameter:', cut_shape, file=sys.stderr)
        return None, None, None
        
    # Incorrect shape verification
    if im.shape[0] != im.shape[1]:
        print('[err] Invalid Shape on image', im.shape, file=sys.stderr)
        return None, None, None # In case of an error, all returned values are None

    try:
        if len(im.shape) == 3:
            im = rgb2gray(im)
    except ValueError:
        print('[err]rgb2gray on image:', image, im.shape, file=sys.stderr)
        return None, None, None # In case of an error, all returned values are None

    im_nse = random_noise(im, var = sig**2, rng = seed)

    # estimate the noise standard deviation from the noisy image
    sigma_est = np.mean(estimate_sigma(im_nse, channel_axis=-1))

    return im, im_nse, sigma_est


def image_metrics(im, im_fil):
    """Compare two images and returns the PSNR, SSIM and MSE of them.
    """
    psnr_value = psnr(im, im_fil, data_range=im_fil.max() - im_fil.min())
    ssim_value = ssim(im, im_fil, data_range=im_fil.max() - im_fil.min())
    mse_value = mse(im, im_fil)
    return psnr_value, ssim_value, mse_value


def rm_tmp_files():
    """Remove temporary files created by UDLF.
    """
    tmp_files = ['input.txt', 'list.txt', 'log.txt', 'output.txt']
    for tmp_file in tmp_files:
        try:
            os.remove(tmp_file)
        except FileNotFoundError:
            pass
