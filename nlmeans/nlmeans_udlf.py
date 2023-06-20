import numpy as np
from .fourier_center import fourier_center
from pyUDLF import run_calls as udlf
from pyUDLF.utils import inputType

import os

def nlmeans_udlf(ima_nse, hW, hP, tau, sig, shape):
    # This is a simple implementation of NL means:
    #
    #   Buades, A. and Coll, B. and Morel, J.M.,
    #
    #   Computer Vision and Pattern Recognition, 2005. CVPR
    #
    # It uses UDLF to define distance between patches

    # UDLF configuration

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

    # UDLF configuration
    input_data = udlf_config(dataset_size=M)
    patch_names = np.reshape(np.arange(0, M, dtype=int), (M, 1))
    np.savetxt('list.txt', patch_names, fmt='%d', delimiter=' ', newline='\n')

    # Main loop
    sum_w = np.zeros((M, N))
    sum_wI = np.zeros((M, N))
    for dx in range(-hW, hW+1):
        for dy in range(-hW, hW+1):
            # Restrict the search window to avoid the central pixel
            if (dx == 0 and dy == 0):
                continue
            # Restrict the search window to be circular
            # if the disk shape is choose
            if (shape == 'disk') and dx**2 + dy**2 > hW**2:
                    continue

            x2range = np.mod(np.arange(0, M) + dx - 1, M)
            y2range = np.mod(np.arange(0, N) + dy - 1, N)

            # Calculate the Euclidean distance between all pairs of
            # patches in the direction (dx, dy)
            diff = (ima_nse - ima_nse[x2range-1, y2range-1])**2
            diff = np.real(np.fft.ifft2(patch_shape * np.fft.fft2(diff)))
            
            # Create the input.txt file for UDLF
            np.savetxt('input.txt', diff, delimiter=' ', newline='\n')
            
            # Run the UDLF framework to get a better distance value
            udlf.run(input_data, get_output=True)
            diff2 = np.loadtxt('output.txt', dtype=np.float64,
                              delimiter=' ', usecols=range(diff.shape[1]))

            if((diff2 - diff) == 0).all():
                print("diff = diff2")

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

    
            pass

    return ima_fil


def udlf_config(dataset_size):
    # Set the paths for UDLF
    udlf.setBinaryPath(os.path.join(os.path.dirname(__file__),
                                    '../udlf/udlf'))
    udlf.setConfigPath(os.path.join(os.path.dirname(__file__),
                                    '../udlf/config.ini'))

    # Set UDLF configuration options properly
    input_data = inputType.InputType()

    # Input dataset files
    input_data.set_param('UDL_TASK', 'UDL')
    input_data.set_param('UDL_METHOD', 'LHRR')
    # input_data.set_param('UDL_METHOD', 'NONE')
    input_data.set_param('SIZE_DATASET', f'{dataset_size}')
    input_data.set_param('INPUT_FILE_FORMAT', 'MATRIX')
    input_data.set_param('INPUT_MATRIX_TYPE', 'DIST')
    input_data.set_param('MATRIX_TO_RK_SORTING', 'HEAP')
    input_data.set_param('INPUT_FILE', 'input.txt')
    input_data.set_param('INPUT_FILE_LIST', 'list.txt')
   
    # Output file settings
    input_data.set_param('OUTPUT_FILE', 'TRUE')
    # input_data.set_param('OUTPUT_FILE_FORMAT', 'RK')
    # input_data.set_param('OUTPUT_RK_FORMAT', 'NUM')
    input_data.set_param('OUTPUT_FILE_FORMAT', 'MATRIX')
    input_data.set_param('OUTPUT_MATRIX_TYPE', 'DIST')
    input_data.set_param('OUTPUT_FILE_PATH', 'output')

    # Evaluation settings
    input_data.set_param('EFFECTIVENESS_EVAL', 'FALSE')
    
    # Method parameters
    input_data.set_param('PARAM_LHRR_K', '18')
    input_data.set_param('PARAM_LHRR_L', f'{dataset_size}')
    input_data.set_param('PARAM_LHRR_T', '2')

    return input_data
