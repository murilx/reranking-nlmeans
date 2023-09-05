import numpy as np
from .fourier_center import fourier_center
from pyUDLF import run_calls as udlf
from pyUDLF.utils import inputType

import os

def nlmeans_udlf(ima_nse, hW, hP, tau, sig, shape, udl_method, udl_params, n_w):
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

    patch_shape = np.zeros((M, N))
    if(shape == 'square'):
        patch_shape = (np.abs(Y - cM) <= hP/2) & (np.abs(X - cN) <= hP/2)
    elif(shape == 'disk'):
        patch_shape = ((Y - cM)**2 + (X - cN)**2) <= hP**2
    patch_shape = patch_shape / np.sum(patch_shape)
    patch_shape = np.conj(np.fft.fft2(np.fft.fftshift(patch_shape)))

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

            x2range = np.mod(np.arange(0, M) + dx, M)
            y2range = np.mod(np.arange(0, N) + dy, N)
            x_idx, y_idx = np.meshgrid(x2range, y2range, indexing='ij')

            # Calculate the Euclidean distance between all pairs of
            # patches in the direction (dx, dy)
            diff = (ima_nse - ima_nse[x_idx, y_idx])**2
            diff = np.real(np.fft.ifft2(patch_shape * np.fft.fft2(diff)))

            # Convert the distance to weights using an exponential
            # kernel (this is a critical step!)
            w = np.exp(- diff / tau**2)

            # Save the weight matrix and its identifiers
            w_values.append(w)
            w_names.append(np.ravel_multi_index([x_idx, y_idx], (M, N)))

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

    # Creation of the weight names list
    weight_names_list = np.reshape(np.arange(0, M * N, dtype=int), (M * N, 1))
    np.savetxt('list.txt', weight_names_list, fmt='%d', delimiter=' ', newline='\n')

    # UDLF configuration
    input_data = udlf_config(
        size_dataset = M*N,
        L = NEIGHBOURHOOD_SIZE,
        udl_method = udl_method,
        udl_params = udl_params
    )

    # Run the UDLF framework to get a ranked list of weights
    udlf.run(input_data, get_output=True)
    new_ranked_lists = np.loadtxt('output.txt',
                                  dtype=int,
                                  delimiter=' ',
                                  usecols=range(ranked_lists.shape[1]))

    sum_w = np.zeros((M, N))
    sum_wI = np.zeros((M, N))
    if n_w is None or num_weights > NEIGHBOURHOOD_SIZE:
        n_w = new_ranked_lists.shape[1]
    for col in range(n_w):
        # Get the `num_weights` first elements of the ranked list array at `pos`
        new_w_names = new_ranked_lists[:, col].reshape(M, N)

        # Get the image coordinates giving the ranked list values
        x_idx, y_idx = np.unravel_index(new_w_names, (M, N))

        # Calculate the Euclidean distance between all pairs of
        # patches in the direction (dx, dy)
        diff = (ima_nse - ima_nse[x_idx, y_idx])**2
        diff = np.real(np.fft.ifft2(patch_shape * np.fft.fft2(diff)))

        # Convert the distance to weights using an exponential
        # kernel (this is a critical step!)
        w = np.exp(- diff / tau**2)

        # Calculate the desnoised value of each pixel
        sum_wI += ima_nse[x_idx, y_idx] * w
        sum_w += w

    ima_fil = sum_wI / sum_w
    return ima_fil


def udlf_config(size_dataset, L, udl_method, udl_params):
    # Set the paths for UDLF
    udlf.setBinaryPath(os.path.join(os.path.dirname(__file__),
                                    '../udlf/udlf'))
    udlf.setConfigPath(os.path.join(os.path.dirname(__file__),
                                    '../udlf/config.ini'))

    # Set UDLF configuration options properly
    input_data = inputType.InputType()

    # Input dataset files
    input_data.set_param('UDL_TASK', 'UDL')
    input_data.set_param('UDL_METHOD', f'{udl_method}')
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

    # Method parameters
    if(udl_method == 'NONE'):
        input_data.set_param('PARAM_NONE_L', f'{L}')

    elif(udl_method == 'CPRR'):
        input_data.set_param('PARAM_CPRR_K', f'{udl_params["k"]}')
        input_data.set_param('PARAM_CPRR_L', f'{L}')
        input_data.set_param('PARAM_CPRR_T', f'{udl_params["t"]}')

    elif(udl_method == 'RLRECOM'):
        input_data.set_param('PARAM_RLRECOM_L', f'{L}')
        input_data.set_param('PARAM_RLRECOM_K', f'{udl_params["k"]}')
        input_data.set_param('PARAM_RLRECOM_LAMBDA', f'{udl_params["lambda"]}')
        input_data.set_param('PARAM_RLRECOM_EPSILON', f'{udl_params["epsilon"]}')

    elif(udl_method == 'RLSIM'):
        input_data.set_param('PARAM_RLSIM_TOPK', f'{udl_params["topk"]}')
        input_data.set_param('PARAM_RLSIM_CK', f'{udl_params["ck"]}')
        input_data.set_param('PARAM_RLSIM_T', f'{udl_params["t"]}')
        input_data.set_param('PARAM_RLSIM_METRIC', f'{udl_params["metric"]}')

    elif(udl_method == 'CONTEXTRR'):
        input_data.set_param('PARAM_CONTEXTRR_L', f'{udl_params["l"]}')
        input_data.set_param('PARAM_CONTEXTRR_K', f'{udl_params["k"]}')
        input_data.set_param('PARAM_CONTEXTRR_T', f'{udl_params["t"]}')
        input_data.set_param('PARAM_CONTEXTRR_NBYK', f'{udl_params["nbyk"]}')
        input_data.set_param('PARAM_CONTEXTRR_OPTIMIZATIONS', f'{udl_params["opt"]}')

    elif(udl_method == 'RECKNNGRAPH'):
        input_data.set_param('PARAM_RECKNNGRAPH_L', f'{L}')
        input_data.set_param('PARAM_RECKNNGRAPH_K', f'{udl_params["k"]}')
        input_data.set_param('PARAM_RECKNNGRAPH_EPSILON', f'{udl_params["epsilon"]}')

    elif(udl_method == 'RKGRAPH'):
        input_data.set_param('PARAM_RKGRAPH_K', f'{udl_params["k"]}')
        input_data.set_param('PARAM_RKGRAPH_T', f'{udl_params["t"]}')
        input_data.set_param('PARAM_RKGRAPH_P', f'{udl_params["p"]}')
        input_data.set_param('PARAM_RKGRAPH_L', f'{L}')

    elif(udl_method == 'CORGRAPH'):
        input_data.set_param('PARAM_CORGRAPH_L', f'{L}')
        input_data.set_param('PARAM_CORGRAPH_K', f'{udl_params["k"]}')
        input_data.set_param('PARAM_CORGRAPH_THRESHOLD_START', f'{udl_params["thold_s"]}')
        input_data.set_param('PARAM_CORGRAPH_THRESHOLD_END', f'{udl_params["thold_e"]}')
        input_data.set_param('PARAM_CORGRAPH_THRESHOLD_INC', f'{udl_params["thold_i"]}')
        input_data.set_param('PARAM_CORGRAPH_CORRELATION', f'{udl_params["corr"]}')

    elif(udl_method == 'LHRR'):
        input_data.set_param('PARAM_LHRR_K', f'{udl_params["k"]}')
        input_data.set_param('PARAM_LHRR_L', f'{L}')
        input_data.set_param('PARAM_LHRR_T', f'{udl_params["t"]}')

    elif(udl_method == 'BFSTREE'):
        input_data.set_param('PARAM_BFSTREE_L', f'{L}')
        input_data.set_param('PARAM_BFSTREE_K', f'{udl_params["k"]}')
        input_data.set_param('PARAM_BFSTREE_CORRELATION_METRIC', f'{udl_params["corr"]}')

    elif(udl_method == 'RDPAC'):
        input_data.set_param('PARAM_RDPAC_K_END', f'{udl_params["k_e"]}')
        input_data.set_param('PARAM_RDPAC_K_INC', f'{udl_params["k_i"]}')
        input_data.set_param('PARAM_RDPAC_K_START', f'{udl_params["k_s"]}')
        input_data.set_param('PARAM_RDPAC_L', f'{L}')
        input_data.set_param('PARAM_RDPAC_L_MULT', f'{udl_params["l_mult"]}')
        input_data.set_param('PARAM_RDPAC_P', f'{udl_params["p"]}')
        input_data.set_param('PARAM_RDPAC_PL', f'{udl_params["pl"]}')

    elif(udl_method == 'RFE'):
        input_data.set_param('PARAM_RFE_K', f'{udl_params[""]}')
        input_data.set_param('PARAM_RFE_T', f'{udl_params[""]}')
        input_data.set_param('PARAM_RFE_L', f'{L}')
        input_data.set_param('PARAM_RFE_PA', f'{udl_params[""]}')
        input_data.set_param('PARAM_RFE_TH_CC', f'{udl_params[""]}')
        input_data.set_param('PARAM_RFE_RERANK_BY_EMB', f'{udl_params[""]}')
        input_data.set_param('PARAM_RFE_EXPORT_EMBEDDINGS', f'{udl_params[""]}')
        input_data.set_param('PARAM_RFE_PERFORM_CSS', f'{udl_params[""]}')
        input_data.set_param('PARAM_RFE_EMBEDDINGS_PATH', f'{udl_params[""]}')
        input_data.set_param('PARAM_RFE_CCS_PATH', f'{udl_params[""]}')

    return input_data
