import numpy as np

def uniform_marginal_log_prior(global_params):
    return -np.inf if np.any(np.array([
        global_params['sigma_int2'],
        global_params['sigmac_int2'],
        global_params['sigmax2'],
        global_params['tau']
    ]) <= 0) else 0