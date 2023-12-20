# Parameters
hW = 10        # Window Size
hP = 3         # Patch Size
tau = 0.15     # Contribution of similarity on noisy-data
sig = 0.1      # Standard deviation of the gaussian noise
shape = 'disk' # Patch shape
n_w = None     # Number of weights to use or use all(None)


image = 'astronaut'  # Image to test with
seed = 42            # Seed to be used when adding noise to the image or None


# UDLF Method and its parameters
udl_method = 'NONE'

def get_udl_params(udl_method):
    return {
        'CPRR':    {'k':3, 't': 1},
        'RLRECOM': {'k':3, 'lambda': 9, 'epsilon': 0.0125},
    }.get(udl_method, None)
