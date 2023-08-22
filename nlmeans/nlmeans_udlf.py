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
    RESEARCH_AREA = (2*hW + 1)**2

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
    w_values = []
    w_names = []

    # Main loop
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
            diff = (ima_nse - ima_nse[x2range, y2range])**2
            diff = np.real(np.fft.ifft2(patch_shape * np.fft.fft2(diff)))
            
            # Convert the distance to weights using an exponential
            # kernel (this is a critical step!)
            w = np.exp(- diff / tau**2)

            # Save the weight matrix and its identifiers
            w_values.append(w)
            w_names.append(np.ones((M, N)) *
                           np.ravel_multi_index([x2range, y2range], (M, N)))
            
    # For the central weight we follow the idea of:
    #   "On two parameters for denoising with Non-Local Means"
    #   J. Salmon, IEEE Signal Process. Lett., 2010
    w_values.append(np.zeros((M, N)) + np.exp(-2*sig**2/tau**2))
    w_names.append(np.arange(M * N).reshape(M, N))

    # Transform the python lists of matrices into a 3D numpy array
    w_values = np.stack(w_values, axis=-1)
    w_names = np.stack(w_names, axis=-1).astype(int)
    NEIGHBOURHOOD_SIZE = w_values.shape[2]

    # Create the ranked list of weight matrices for udlf
    ranked_lists = np.zeros((M * N, NEIGHBOURHOOD_SIZE), dtype=int)
    for i in range(M):
        for j in range(N):
            rl = np.rec.fromarrays((w_names[i, j, :], w_values[i, j, :]),
                                   names=('names', 'values'))
            rl = rl[rl['values'].argsort()]
            ranked_lists[i * M + j, :] = np.copy(rl['names'])

    # Create the input file for the UDLF
    np.savetxt('input.txt', ranked_lists, fmt='%d', delimiter=' ', newline='\n')

    # Run the UDLF framework to get a ranked list of weights
    udlf.run(input_data, get_output=True)
    new_ranked_lists = np.loadtxt('output.txt',
                                  dtype=int,
                                  delimiter=' ',
                                  usecols=range(ranked_lists.shape[1]))

    sum_w = np.zeros((M, N))
    sum_wI = np.zeros((M, N))
    for pos in range(new_ranked_lists.shape[0]):
        # Get weight coordinates giving the ranked list position
        wx, wy = np.unravel_index(pos, (M, N))

        # Get the image coordinates giving the ranked list values
        ix, iy = np.unravel_index(new_ranked_lists[pos, :num_weights], (M, N))

        # Get the indices of every weight
        # excluding the last `num_weights` of the list
        new_w_names = new_ranked_lists[pos, :num_weights]
        w_index = w_names[wx, wy, :].argsort()[new_w_names.argsort().argsort()]
        
        # Calculate the desnoised value of each pixel
        sum_wI[wx, wy] = np.sum(ima_nse[ix, iy] * w_values[wx, wy, w_index])
        sum_w[wx, wy] = np.sum(w_values[wx, wy, w_index])
    
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
    input_data.set_param('UDL_METHOD', 'LHRR')
    input_data.set_param('SIZE_DATASET', f'{size_dataset}')
    input_data.set_param('INPUT_FILE_FORMAT', 'RK')
    input_data.set_param('INPUT_RK_FORMAT', 'NUM')
    input_data.set_param('INPUT_FILE', 'input.txt')
    input_data.set_param('INPUT_FILE_LIST', 'list.txt')
   
    # Output file settings
    input_data.set_param('OUTPUT_FILE', 'TRUE')
    input_data.set_param('OUTPUT_FILE_FORMAT', 'RK')
    input_data.set_param('OUTPUT_RK_FORMAT', 'NUM')
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
