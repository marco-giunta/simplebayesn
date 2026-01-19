import numpy as np
from scipy.special import log_ndtr
from ..utils.intrinsic import get_mean_int, get_cov_int
from ..utils.data import SaltData, SaltDataCompact
from scipy.special import logsumexp

def marginal_loglikelihood(global_params: dict,
                           observed_data: SaltData | SaltDataCompact):
    
    mean_int = get_mean_int(global_params).flatten()
    cov_int = get_cov_int(global_params)
    e1 = np.array([1, 0, 0])

    Sigma_inv = np.linalg.inv(
        cov_int + observed_data.cov +
        observed_data.sigma_mu_z2[:, np.newaxis, np.newaxis] * np.outer(e1, e1)
    )

    eE = np.array([global_params['RB'], 1, 0])
    sE = 1 / np.sqrt(
        np.einsum('i,nij,j->n', eE, Sigma_inv, eE)
    )
    d = np.column_stack((observed_data.m_app, observed_data.c_app, observed_data.x))
    y = d - mean_int - observed_data.dist_mod[:, np.newaxis] * e1 # np.outer(observed_data.dist_mod, e1)
    # np.column_stack([observed_data.dist_mod, np.zeros_like(observed_data.dist_mod), np.zeros_like(observed_data.dist_mod)])

    E_hat = sE**2 * np.einsum('i,nij,nj->n', eE, Sigma_inv, y)

    log_prefactor = np.log(sE * np.sqrt(2 * np.pi))
    v = y - E_hat[:, np.newaxis] * eE
    log_norm_factor = -0.5 * (
        np.einsum('ni,nij,nj->n', v, Sigma_inv, v) +
        # np.log((2*np.pi)**3 / np.linalg.det(Sigma_inv))
        3 * np.log(2 * np.pi) - np.linalg.slogdet(Sigma_inv)[1]
    )
    log_exp_factor = -np.log(global_params['tau']) + (0.5 * (sE/global_params['tau']) ** 2 - E_hat/global_params['tau'])
    log_cdf_factor = log_ndtr(E_hat/sE - sE/global_params['tau'])
    result = log_prefactor + log_norm_factor + log_exp_factor + log_cdf_factor

    return result.sum()
