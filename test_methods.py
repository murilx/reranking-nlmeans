from time import time

from nlmeans.nlmeans import nlmeans
from nlmeans.nlmeans_udlf import nlmeans_udlf

from parameters import *
from utils import *

im, im_name = get_image(image)
im, im_nse, sigma_est = process_image(im, sig, seed, cut_shapes[im_name])


udl_param_dict = {
    'RLSIM': {'topk': 5, 't': 3, 'metric': 'RBO'},
    'CONTEXTRR': {'k': 5, 't': 3, 'nbyk': 1, 'opt': 'FALSE'},
    'RECKNNGRAPH': {'k': 5, 'epsilon': 0.0125},
    'RKGRAPH': {'k': 5, 't': 1, 'p': 0.95},
    'CORGRAPH': {'k': 5, 'thold_start': 0.35, 'thold_end': 1, 'thold_inc': 0.005, 'corr': 'PEARSON'},
    'LHRR': {'k': 3, 't': 2},
    'BFSTREE': {'k': 5, 'corr': 'RBO'}
}

for udl_method, udl_params in udl_param_dict.items():
    try:
        print('Starting', udl_method)
        time_udlf = time()
        im_fil2 = nlmeans_udlf(im_nse, hW, hP, tau, sigma_est, shape, udl_method, udl_params, n_w)
        print(f'time spent on "{udl_method}": {time() - time_udlf}')
        rm_tmp_files()
    except FileNotFoundError:
        print('ERROR:', udl_method)
