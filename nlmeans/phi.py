import numpy as np

def phi(d):
    if (d < 1).all():
        w = 1
    else:
        w = np.exp(-(d-1)/4)
    return w
