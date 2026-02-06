import numpy as np
from scipy.stats import invgamma

def uniform_marginal_log_prior(global_params):
    return -np.inf if np.any(np.array([
        global_params['sigma_int2'],
        global_params['sigmac_int2'],
        global_params['sigmax2'],
        global_params['tau']
    ]) <= 0) else 0

def uniform_marginal_log_prior_invgamma_sigmac_int2(global_params, alpha: float = 0.003, beta: float = 0.003):
    return -np.inf if np.any(np.array([
        global_params['sigma_int2'],
        global_params['sigmax2'],
        global_params['tau']
    ]) <= 0) else invgamma.logpdf(global_params['sigmac_int2'], a=alpha, scale=beta)