import numpy as np
from .fourier_center import fourier_center

def nlmeans(ima_nse, hW, hP, tau, sig, shape):
    # This is a simple implementation of NL means:
    #
    #   Buades, A. and Coll, B. and Morel, J.M.,
    #
    #   Computer Vision and Pattern Recognition, 2005. CVPR
    #
    # It uses the fast FFT-based algorithm as described in:
    #
    #   Charles-Alban Deledalle, Vincent Duval and Joseph Salmon,
    #   "Non-Local Methods with Shape-Adaptive Patches (NLM-SAP)",
    #   Journal of Mathematical Imaging and Vision, pp. 1-18, 2011
    #
    # Author: Charles Deledalle
    
    # Define a patch shape in the Fourier domain
    M, N = ima_nse.shape
    cM, cN = fourier_center(M, N)
    Y, X = np.meshgrid(np.arange(0, M), np.arange(0, N))

    patch_shape = np.zeros((M, N))
    if(shape == 'square'):
        patch_shape = (np.abs(Y - cM) <= hP/2) & (np.abs(X - cN) <= hP/2)
    elif(shape == 'disk'):
        patch_shape = ((Y - cM)**2 + (X - cN)**2) <= hP**2
    patch_shape = patch_shape / np.sum(patch_shape)
    patch_shape = np.conj(np.fft.fft2(np.fft.fftshift(patch_shape)))

    # Main loop
    sum_w = np.zeros((M, N))
    sum_wI = np.zeros((M, N))
    for dx in range(-hW, hW+1):
        for dy in range(-hW, hW+1):
            # Restrict the search window to be circular
            # and avoid the central pixel
            # if (dx == 0 and dy == 0) or dx**2 + dy**2 > hW**2:
            #     continue
            x2range = np.mod(np.arange(0, M) + dx - 1, M)
            y2range = np.mod(np.arange(0, N) + dy - 1, N)

            # Calculate the Euclidean distance between all pairs of
            # patches in the direction (dx, dy)
            diff = (ima_nse - ima_nse[x2range-1, y2range-1])**2
            diff = np.real(np.fft.ifft2(patch_shape * np.fft.fft2(diff)))

            # Convert the distance to weights using an exponential
            # kernel (this is a critical step!)
            w = np.exp(- diff / tau**2)

            # Increment accumulators for the weighted average
            sum_w += w
            sum_wI += w * ima_nse[x2range-1, y2range-1]

    # For the central weight we follow the idea of:
    #   "On two parameters for denoising with Non-Local Means"
    #   J. Salmon, IEEE Signal Process. Lett., 2010
    sum_w += np.exp(-2*sig**2/tau**2)
    sum_wI += np.exp(-2*sig**2/tau**2) * ima_nse

    # Weighted average
    ima_fil = sum_wI / sum_w
    # ima_w = w / sum_w

    return ima_fil
