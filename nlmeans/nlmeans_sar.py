import numpy as np
from nlmeans_sar_it import nlmeans_sar_it

def nlmeans_sar(ima_nse, hW, hP, tau):
    # This realizes an NL means scheme for Nakagami-Rayleigh
    # distribution following (cf also PPB non iteratif):
    #
    #    Charles-Alban Deledalle, Loïc Denis and Florence Tupin,
    #    "Iterative Weighted Maximum Likelihood Denoising with Probabilistic Patch-Based Weights",
    #    IEEE Trans. on Image Processing, vol. 18, no. 12, pp. 2661-2672, December 2009
    #
    # Author: Charles Deledalle
    
    ima_fil = nlmeans_sar_it(ima_nse, np.ones_like(ima_nse),
                             hW, hP,
                             tau, 1)
    
    return ima_fil
