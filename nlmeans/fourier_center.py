def fourier_center(M, N):
    # Location of the central point of the kernel for building the shapes
    # INPUT:
    #    M, N  : image size
    # OUTPUT:
    #   cM,cN  : center position of the kernel
    #
    # Authors: Deledalle, Duval, Salmon
    
    # WARNING: +1 shift due to Matlab's convention for FFT
    
    if M % 2 == 1:
        cM = (M + 3) // 2
    else:
        cM = (M + 2) // 2
    
    if N % 2 == 1:
        cN = (N + 3) // 2
    else:
        cN = (N + 2) // 2
    
    return cM, cN
