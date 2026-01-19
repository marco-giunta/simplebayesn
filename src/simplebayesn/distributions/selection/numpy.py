import numpy as np
from ...utils.data import SaltData, SaltDataCompact
from ...utils.intrinsic import get_mean_int, get_cov_int
from scipy.special import log_ndtr, logsumexp

def log_selection_probability(global_params: dict,
                              observed_data: SaltData | SaltDataCompact,
                              mlim: tuple[float], clim: tuple[float], xlim: tuple[float],
                              Nm: int, Nc: int, Nx: int):
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
    ) # (N_SN, )
    sE_exp = np.expand_dims(sE, axis=(1, 2, 3))
    del sE

    # np.array(np.meshgrid(...)) == np.stack(np.meshgrid(...), axis=0)
    N_SN = observed_data.num_samples
    volume_element = (np.diff(mlim) * np.diff(clim) * np.diff(xlim) / ((Nm - 1) * (Nc - 1) * (Nx - 1)))[0]

    d = np.repeat(
        np.stack(
            np.meshgrid(
                np.linspace(*mlim, num=Nm),
                np.linspace(*clim, num=Nc),
                np.linspace(*xlim, num=Nx)
            ),
            axis=0
        )[np.newaxis, ...],
        repeats=N_SN, axis=0
    ) # (N_SN, 3, Nm, Nc, Nx)

    dm = np.zeros_like(d)
    dm[:, 0] = observed_data.dist_mod[:, np.newaxis, np.newaxis, np.newaxis]
    # observed_data.dist_mod[:, np.newaxis, np.newaxis, np.newaxis] == np.expand_dims(observed_data.dist_mod, axis=(1, 2, 3))
    mi = mean_int[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
    # mean_int[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis] == np.expand_dims(mean_int, axis=(0, 2, 3))

    y = d - mi - dm # (N_SN, 3, Nm, Nc, Nx)
    del d, mi, dm

    E_hat = sE_exp ** 2 * \
          np.einsum('i,nij,njklm->nklm', eE, Sigma_inv, y)
    # (N_SN, Nm, Nc, Nx)

    log_prefactor = np.log(sE_exp * np.sqrt(2 * np.pi))

    EE = np.zeros_like(y)
    EE[:, 0] = global_params['RB'] * E_hat
    EE[:, 1] = E_hat
    v = y - EE
    del y, EE
    # log_norm_factor = -0.5 * (np.einsum(
    #     'niklm,nij,njklm->nklm',
    #     v, Sigma_inv, v
    # ) + np.log(np.expand_dims(
    #     ((2*np.pi)**3 / np.linalg.det(Sigma_inv)),
    #     axis=(1, 2, 3)
    # ))) # (N_SN, Nm, Nc, Nx)
    log_norm_factor = -0.5 * (np.einsum('niklm,nij,njklm->nklm', v, Sigma_inv, v)
                              + np.log(np.expand_dims(((2 * np.pi) ** 3 / np.linalg.det(Sigma_inv)), axis=(1, 2, 3))))
    del v

    log_exp_factor = -np.log(global_params['tau']) + (
        0.5 * \
            (sE_exp / global_params['tau']) ** 2 \
        - E_hat/global_params['tau']
    )
    log_cdf_factor = log_ndtr(
        E_hat / sE_exp \
        - sE_exp / global_params['tau']
    )
    del E_hat, sE_exp

    ll_grid = log_prefactor + log_norm_factor + log_exp_factor + log_cdf_factor
    del log_prefactor, log_norm_factor, log_exp_factor, log_cdf_factor
    
    max_ll = ll_grid.max(axis=(1, 2, 3))
    # integrals = (np.exp(ll_grid - max_ll[:, None, None, None]).sum(axis=(1, 2, 3)) * volume_element) * np.exp(max_ll)
    # return np.prod(integrals)
    log_integrals = logsumexp(ll_grid - max_ll[:, None, None, None], axis=(1, 2, 3)) + np.log(volume_element) + max_ll
    return log_integrals.sum()