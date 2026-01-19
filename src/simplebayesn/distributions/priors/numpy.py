import numpy as np

def get_priors_params_uniform_priors():
    return {
        'tau':{'alpha_prior':-1, 'beta_prior':0},
        'RB': {'mean_prior':None, 'std_prior':None},
        'x':  {'alpha_prior':-1, 'beta_prior':0, 'inv_cov_prior':None, 'mean_prior':np.array([0])},
        'c':  {'alpha_prior':-1, 'beta_prior':0, 'inv_cov_prior':None, 'mean_prior':np.array([0, 0])},
        'M':  {'alpha_prior':-1, 'beta_prior':0, 'inv_cov_prior':None, 'mean_prior':np.array([0, 0, 0])},
    }