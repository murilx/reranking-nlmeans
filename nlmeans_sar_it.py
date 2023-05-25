import numpy as np
from fourier_center import fourier_center
from phi import phi

def nlmeans_sar_it(ima_nse, ima_est, hW, hP, tau, T):
    # Force image to be defined on R^+
    ima_nse[ima_nse <= 0] = np.min(ima_nse[ima_nse > 0])

    # Define a patch shape in the fourier domain
    M, N = ima_nse.shape
    cM, cN = fourier_center(M, N)
    patch_shape = np.zeros((M, N))
    Y, X = np.meshgrid(np.arange(1, M+1), np.arange(1, N+1))
    patch_shape = (Y - cM)**2 + (X - cN)**2 <= hP**2
    patch_shape = patch_shape / np.sum(patch_shape)
    patch_shape = np.conj(np.fft.fft2(np.fft.fftshift(patch_shape)))

    # Main loop
    sum_w = np.zeros((M, N))
    sum_wI = np.zeros((M, N))
    for dx in range(-hW, hW+1):
        for dy in range(-hW, hW+1):
            # Restrict the search window to be circular
            # and avoid the central pixel
            if (dx == 0 and dy == 0) or dx**2 + dy**2 > hW**2:
                continue
            x2range = np.mod(np.arange(0, M) + dx - 1, M) 
            y2range = np.mod(np.arange(0, N) + dy - 1, N)
            # x2range = np.mod(np.arange(1, M+1) + dx - 1, M) + 1 
            # y2range = np.mod(np.arange(1, N+1) + dy - 1, N) + 1

            # Calculate the generalized likelihood ratio based dissimilarity
            # derived for Nakagami-Rayleigh distributions.
            diff = np.log(np.divide(ima_nse, ima_nse[x2range[:, None], y2range[None, :]]) + \
                          np.divide(ima_nse[x2range[:, None], y2range[None, :]], ima_nse)) - \
                   np.log(2)
            # diff = np.log(ima_nse / ima_nse[x2range.flatten(), y2range.flatten()] + \
            #               ima_nse[x2range.flatten(), y2range.flatten()] / ima_nse) - \
            #        np.log(2)
            diff = np.real(np.fft.ifft2((patch_shape * np.fft.fft2(diff))))

            # Calculate the Kullback-Leibler divergence based dissimilarity
            # derived for Nakagami-Rayleigh distributions.
            diff2 = np.divide(ima_est**2, ima_est[x2range[:, None], y2range[None, :]]**2) + \
                    np.divide(ima_est[x2range[:, None], y2range[None, :]]**2, ima_est**2) - \
                   2
            # diff2 = ima_est**2 / ima_est[x2range.flatten(), y2range.flatten()]**2 + \
            #         ima_est[x2range.flatten(), y2range.flatten()]**2 / ima_est**2 - \
            #        2
            diff2 = np.real(np.fft.ifft2((patch_shape * np.fft.fft2(diff2))))

            # Combine both dissimilarity criteria and convert them
            # to weights using a kernel suitable for correlated
            # speckle (this is a critical step!)
            w = phi(diff / tau**2 + diff2 / T**2)

            # Increment accumulators for the weighted maximum
            # likelihood estimation. The sum of the square of the
            # amplitude according to is related to the
            # Nakagami-Rayleigh distributions.
            sum_w += w
            sum_wI += w * ima_nse[x2range[:, None], y2range[None, :]]**2

    # For the central weight we follow the idea of:
    #   "On two parameters for denoising with Non-Local Means"
    #   J. Salmon, IEEE Signal Process. Lett., 2010
    sum_w += np.exp(-2 * (1 - np.sqrt(ima_est / np.max(ima_est)))**2 / tau**2)
    sum_wI += sum_w * ima_nse**2

    # Compute the final estimate
    ima_den = sum_wI / (sum_w + (sum_w == 0))

    return ima_den
