import numpy as np

def get_priors_params_uniform_priors():
    return {
        'tau':{'alpha_prior':-1, 'beta_prior':0},
        'RB': {'mean_prior':None, 'std_prior':None},
        'x':  {'alpha_prior':-1, 'beta_prior':0, 'inv_cov_prior':None, 'mean_prior':np.array([0])},
        'c':  {'alpha_prior':-1, 'beta_prior':0, 'inv_cov_prior':None, 'mean_prior':np.array([0, 0])},
        'M':  {'alpha_prior':-1, 'beta_prior':0, 'inv_cov_prior':None, 'mean_prior':np.array([0, 0, 0])},
    }

def get_priors_params_uniform_priors_invgamma_sigmac_int2(alpha: float = 0.003, beta: float = 0.003):
    prior_params_dict = get_priors_params_uniform_priors()
    prior_params_dict['c']['alpha_prior'] = alpha
    prior_params_dict['c']['beta_prior'] = beta
    return prior_params_dict

def get_priors_params_uniform_invgamma(alpha: float = 0.003, beta: float = 0.003):
    prior_params_dict = get_priors_params_uniform_priors()
    # sigma_int2
    prior_params_dict['M']['alpha_prior'] = alpha
    prior_params_dict['M']['beta_prior'] = beta
    # sigmac_int2
    prior_params_dict['c']['alpha_prior'] = alpha
    prior_params_dict['c']['beta_prior'] = beta
    # sigmax2
    prior_params_dict['x']['alpha_prior'] = alpha
    prior_params_dict['x']['beta_prior'] = beta
    # tau
    prior_params_dict['tau']['alpha_prior'] = alpha
    prior_params_dict['tau']['beta_prior'] = beta
    return prior_params_dict