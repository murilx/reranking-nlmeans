import os
from time import time
import datetime

import matplotlib.pyplot as plt
import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver
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

ex = Experiment()
ex.observers.append(MongoObserver(
    url = 'localhost:27017',
    db_name = 'sacred'
))

# Configs for each UDLF Method
@ex.config
def lhrr():
    """Configuration using LHRR."""
    # Non-Local Means parameters
    hW = 10        # Window Size
    hP = 3         # Patch Size
    tau = 0.15     # Contribution of similarity on noisy-data
    sig = 0.1      # Standard deviation of the gaussian noise
    shape = 'disk' # Patch shape
    n_w = None     # Number of weights to use or use all(None)

    # Image to test with
    image = 'astronaut'

    # UDLF Parameters
    udl_method = 'LHRR'
    udl_params = {
        'k': 18,
        't': 2
    }

@ex.named_config
def none():
    """Configuration using NONE."""
    udl_method = 'NONE'

@ex.named_config
def cprr():
    """Configuration using CPRR."""
    # UDLF Parameters
    udl_method = 'CPRR'
    udl_params = {
        'k': 20,
        't': 2
    }

@ex.named_config
def rlrecom():
    """Configuration using RLRECOM."""
    # UDLF Parameters
    udl_method = 'RLRECOM'
    udl_params = {
        'k': 8,
        'lambda': 2,
        'epsilon': 0.0125
    }

@ex.named_config
def rlsim():
    """Configuration using RLSIM."""
    # UDLF Parameters
    udl_method = 'RLSIM'
    udl_params = {
        'topk': 15,
        't': 3,
        'metric': 'INTERSECTION'
    }

@ex.named_config
def contextrr():
    """Configuration using CONTEXTRR."""
    # UDLF Parameters
    udl_method = 'CONTEXTRR'
    udl_params = {
        'l': 25,
        'k': 7,
        't': 5,
        'nbyk': 1,
        'opt': 'TRUE'
    }

@ex.named_config
def recknngraph():
    """Configuration using RECKNNGRAPH."""
    # UDLF Parameters
    udl_method = 'RECKNNGRAPH'
    udl_params = {
        'k': 15,
        'epsilon': 0.0125
    }

@ex.named_config
def rkgraph():
    """Configuration using RKGRAPH."""
    # UDLF Parameters
    udl_method = 'RKGRAPH'
    udl_params = {
        'k': 20,
        't': 1,
        'p': 0.95
    }

@ex.named_config
def corgraph():
    """Configuration using CORGRAPH."""
    # UDLF Parameters
    udl_method = 'CORGRAPH'
    udl_params = {
        'k': 25,
        'thold_s': 0.35,
        'thold_e': 1,
        'thold_i': 0.005,
        'corr': 'PEARSON'
    }

@ex.named_config
def bfstree():
    """Configuration using BFSTREE."""
    # UDLF Parameters
    udl_method = 'BFSTREE'
    udl_params = {
        'k': 20,
        'corr': 'RBO'
    }

@ex.named_config
def rdpac():
    """Configuration using RDPAC."""
    # UDLF Parameters
    udl_method = 'RDPAC'
    udl_params = {
        'k_e': 15,
        'k_i': 1,
        'k_s': 1,
        'l_mult': 2,
        'p': 0.60,
        'pl': 0.99
    }

@ex.named_config
def rfe():
    """Configuration using RFE."""
    # UDLF Parameters
    udl_method = 'RFE'
    udl_params = {
        'k': 20,
        't': 2,
        'pa': 0.1,
        'th_cc': 0,
        'rr_by_emb': 'FALSE',
        'emb_exp': 'FALSE',
        'css': 'FALSE',
        'emb_path': 'embeddings.txt',
        'css_path': 'css.txt'
    }

@ex.automain
def main(_run, _log, image, hW, hP, tau, sig, shape, n_w, udl_method, udl_params, seed):
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

    # Remove the temporary files created
    tmp_files_created = ['input.txt', 'list.txt', 'log.txt', 'output.txt']
    for tmp_file in tmp_files_created:
        try:
            os.remove(tmp_file)
        except FileNotFoundError:
            pass

    # Save images generated
    fmt_date = datetime.datetime.now()
    fmt_date = fmt_date.strftime('%d-%m-%Y-%H-%M') # day-month-year-hour-minute

    im_nse_file = os.path.join(OUT_DIR, f'{im_name}_noise_{fmt_date}_{seed}.png')
    plt.imsave(im_nse_file, im_nse, cmap='gray')

    nl_file = os.path.join(OUT_DIR, f'{im_name}_nlmeans_{fmt_date}_{seed}.png')
    plt.imsave(nl_file, im_fil1, cmap='gray')

    udlf_file = os.path.join(OUT_DIR, f'{im_name}_udlf-{udl_method}-{shape}_{fmt_date}_{seed}.png')
    plt.imsave(udlf_file, im_fil2, cmap='gray')

    ex.add_artifact(im_nse_file)
    ex.add_artifact(nl_file)
    ex.add_artifact(udlf_file)

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

    # Show information on stdout

    _log.info(f'Seed: {seed}\n')

    _log.info('Noised Image:')
    _log.info(f'PSNR: {im_noise_psnr}')
    _log.info(f'SSIM: {im_noise_ssim}')
    _log.info(f'MSE : {im_noise_mse}\n')

    _log.info('Non-Local Means:')
    _log.info(f'Time: {nlmeans_sap_time}')
    _log.info(f'PSNR: {nlm_noise_psnr}')
    _log.info(f'SSIM: {nlm_noise_ssim}')
    _log.info(f'MSE : {nlm_noise_mse}\n')

    _log.info('UDLF Non-Local Means:')
    _log.info(f'Time: {nlmeans_udlf_time}')
    _log.info(f'PSNR: {udlf_noise_psnr}')
    _log.info(f'SSIM: {udlf_noise_ssim}')
    _log.info(f'MSE : {udlf_noise_mse}\n')

    # Save info
    _run.info = {
        'im_noise_pnsr'  : im_noise_psnr,
        'im_noise_ssim'  : im_noise_ssim,
        'im_noise_mse'   : im_noise_mse,

        'nlmeans_time'   : nlmeans_sap_time,
        'nlm_noise_psnr' : nlm_noise_psnr,
        'nlm_noise_ssim' : nlm_noise_ssim,
        'nlm_noise_mse'  : nlm_noise_mse,

        'nlmeans_udlf_time': nlmeans_udlf_time,
        'udlf_noise_pnsr': udlf_noise_psnr,
        'udlf_noise_ssim': udlf_noise_ssim,
        'udlf_noise_mse' : udlf_noise_mse
    }
