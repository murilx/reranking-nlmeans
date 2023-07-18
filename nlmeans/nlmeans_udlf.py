import numpy as np
from .fourier_center import fourier_center
from pyUDLF import run_calls as udlf
from pyUDLF.utils import inputType

import os

def nlmeans_udlf(ima_nse, hW, hP, tau, sig, shape, num_weights=None):
    # This is a simple implementation of NL means:
    #
    #   Buades, A. and Coll, B. and Morel, J.M.,
    #
    #   Computer Vision and Pattern Recognition, 2005. CVPR
    #
    # It uses UDLF to define distance between patches

    # Define a patch shape in the Fourier domain
    M, N = ima_nse.shape
    cM, cN = fourier_center(M, N)
    Y, X = np.meshgrid(np.arange(0, M), np.arange(0, N))
    RESEARCH_AREA = (2*hW+1)**2

    patch_shape = np.zeros((M, N))
    if(shape == 'square'):
        patch_shape = (np.abs(Y - cM) <= hP/2) & (np.abs(X - cN) <= hP/2)
    elif(shape == 'disk'):
        patch_shape = ((Y - cM)**2 + (X - cN)**2) <= hP**2
    patch_shape = patch_shape / np.sum(patch_shape)
    patch_shape = np.conj(np.fft.fft2(np.fft.fftshift(patch_shape)))

    # UDLF configuration
    input_data = udlf_config(size_dataset=M*N, L=RESEARCH_AREA)

    # Creation of the weight names list
    weight_names_list = np.reshape(np.arange(0, M * N, dtype=int), (M * N, 1))
    np.savetxt('list.txt', weight_names_list, fmt='%d', delimiter=' ', newline='\n')

    # Weight value and weight names matrices
    w_values = np.zeros((M, N, RESEARCH_AREA))
    w_names = np.zeros((M, N, RESEARCH_AREA), dtype=int)
    w_num = 0

    # Main loop
    for dx in range(-hW, hW+1):
        for dy in range(-hW, hW+1):
            x2range = np.mod(np.arange(0, M) + dx - 1, M)
            y2range = np.mod(np.arange(0, N) + dy - 1, N)

            # Restrict the search window to avoid the central pixel
            if (dx == 0 and dy == 0):
                # For the central weight we follow the idea of:
                #   "On two parameters for denoising with Non-Local Means"
                #   J. Salmon, IEEE Signal Process. Lett., 2010
                w = np.ones((M, N)) * np.exp(-2*sig**2/tau**2)
                w_values[:,:,w_num] = w
                w_names[:,:,w_num] = (x2range.reshape((M, 1)) * M +
                                      y2range.reshape((1, N))).astype(int)
                w_num += 1
                continue
            
            # Restrict the search window to be circular
            # if the disk shape is choose
            if (shape == 'disk') and dx**2 + dy**2 > hW**2:
                    continue

            # Calculate the Euclidean distance between all pairs of
            # patches in the direction (dx, dy)
            diff = (ima_nse - ima_nse[x2range, y2range])**2
            diff = np.real(np.fft.ifft2(patch_shape * np.fft.fft2(diff)))
            
            # Convert the distance to weights using an exponential
            # kernel (this is a critical step!)
            w = np.exp(- diff / tau**2)

            # Save the weight matrix and its identifier
            w_values[:,:,w_num] = w
            w_names[:,:,w_num] = x2range.reshape((M, 1)) * M + y2range.reshape((1, N))
            w_num += 1

    # Create the ranked list of weight matrices for udlf
    ranked_lists = np.zeros((M * N, RESEARCH_AREA), dtype=int)
    rl = np.zeros((RESEARCH_AREA, 2), dtype=int)
    for i in range(M):
        for j in range(N):
            rl[:, 0] = np.copy(w_names[i, j, :])
            rl[:, 1] = np.copy(w_values[i, j, :])
            rl = rl[rl[:, 1].argsort()]
            ranked_lists[i * M + j, :] = np.copy(rl[:, 0]) 

    # Create the input file for the UDLF
    np.savetxt('input.txt', ranked_lists, fmt='%d', delimiter=' ', newline='\n')

    # Run the UDLF framework to get a ranked list of weights
    # TODO For the none parameter udlf cannot run and return the message "Killed"
    udlf.run(input_data, get_output=True)
    # new_ranked_lists = np.loadtxt('output.txt',
    #                               dtype=int,
    #                               delimiter=' ',
    #                               usecols=range(ranked_lists.shape[1]))

    # Denoise the image using the new weights based on the UDLF ranked lists
    if num_weights == None:
        num_weights == RESEARCH_AREA
    sum_w = np.zeros((M, N))
    sum_wI = np.zeros((M, N))
    new_ranked_lists = ranked_lists # TEMPORARY for tests only
    for pos in range(new_ranked_lists.shape[0]):
        # Get image coordinates giving the ranked list position
        ix = pos // M
        iy = pos % M

        # Get the indices of every weight
        # excluding the last `num_weights` of the list
        new_w_names = new_ranked_lists[pos, :num_weights]
        weight_indices = np.where(w_names[ix, iy, :num_weights] == new_w_names[:, None])[1]

        # Calculate the desnoised value of each pixel
        sum_wI[ix, iy] = np.sum(ima_nse[ix, iy] * w_values[ix, iy, weight_indices])
        sum_w[ix, iy] = np.sum(w_values[ix, iy, weight_indices])

    ima_fil = sum_wI / sum_w
    return ima_fil


def udlf_config(size_dataset, L):
    # Set the paths for UDLF
    udlf.setBinaryPath(os.path.join(os.path.dirname(__file__),
                                    '../udlf/udlf'))
    udlf.setConfigPath(os.path.join(os.path.dirname(__file__),
                                    '../udlf/config.ini'))

    # Set UDLF configuration options properly
    input_data = inputType.InputType()

    # Input dataset files
    input_data.set_param('UDL_TASK', 'UDL')
    # input_data.set_param('UDL_METHOD', 'LHRR')
    input_data.set_param('UDL_METHOD', 'NONE')
    input_data.set_param('SIZE_DATASET', f'{size_dataset}')
    # input_data.set_param('INPUT_FILE_FORMAT', 'MATRIX')
    # input_data.set_param('INPUT_MATRIX_TYPE', 'DIST')
    # input_data.set_param('MATRIX_TO_RK_SORTING', 'HEAP')
    input_data.set_param('INPUT_FILE_FORMAT', 'RK')
    input_data.set_param('INPUT_RK_FORMAT', 'NUM')
    input_data.set_param('INPUT_FILE', 'input.txt')
    input_data.set_param('INPUT_FILE_LIST', 'list.txt')
   
    # Output file settings
    input_data.set_param('OUTPUT_FILE', 'TRUE')
    input_data.set_param('OUTPUT_FILE_FORMAT', 'RK')
    input_data.set_param('OUTPUT_RK_FORMAT', 'NUM')
    # input_data.set_param('OUTPUT_FILE_FORMAT', 'MATRIX')
    # input_data.set_param('OUTPUT_MATRIX_TYPE', 'DIST')
    input_data.set_param('OUTPUT_FILE_PATH', 'output')

    # Evaluation settings
    input_data.set_param('EFFECTIVENESS_EVAL', 'FALSE')

    # NONE method parameters
    input_data.set_param('PARAM_NONE_L', f'{L}')
    
    # LHRR method parameters
    input_data.set_param('PARAM_LHRR_K', '18')
    input_data.set_param('PARAM_LHRR_L', f'{L}')
    input_data.set_param('PARAM_LHRR_T', '2')

    return input_data
