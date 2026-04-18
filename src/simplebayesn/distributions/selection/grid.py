import numpy as np
from scipy.special import log_ndtr, logsumexp
from ...utils.data import SaltData
from ...utils.intrinsic import get_mean_int, get_cov_int, get_mean_int_vectorized, get_cov_int_vectorized
from ...utils.param_array import from_param_batch
import torch

def log_selection_probability_grid(global_params: dict,
                                   observed_data: SaltData,
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

@torch.no_grad
def log_selection_prob_grid_vectorized(
        global_params_batch: torch.Tensor,
        observed_data: SaltData,
        mlim: tuple[float, float],
        clim: tuple[float, float],
        xlim: tuple[float, float],
        Nm: int,
        Nc: int,
        Nx: int,
        sn_batch_size: int = 32,
        device: torch.device | str = 'cuda',
        dtype: torch.dtype | str = torch.float32
) -> torch.Tensor:
    """
    Compute grid log selection probabilities for W parameter vectors simultaneously.
 
    Evaluates the dust-marginalised log-likelihood on a regular
    ``(Nm, Nc, Nx)`` observable grid for all W walker proposals at once, and
    approximates the selection integral by a stable log-sum-exp Riemann sum.
 
    Unlike the MC estimator, this function is **deterministic** — no random
    draws are made — which eliminates MC noise from the log-posterior and is
    preferable when the observable box boundaries are well-defined.
 
    **Memory management:** The grid itself has ``Nm x Nc x Nx`` points.
    Holding all W x N SN log-likelihood grids simultaneously would require
    O(W x N x Nm x Nc x Nx) memory, which is infeasible for large N or fine
    grids.  Instead, SNe are processed in chunks of ``sn_batch_size``,
    keeping peak memory at O(W x ``sn_batch_size`` x Nm x Nc x Nx).
    The walker dimension W and the grid are fully vectorised within each
    chunk; only the SN loop is explicit.
 
    Parameters
    ----------
    global_params_batch : torch.Tensor or array-like, shape (W, 11)
        W parameter vectors stacked row-wise, in canonical
        :data:`~simplebayesn.utils.param_array.PARAM_KEYS` order.
        Converted internally via ``torch.as_tensor``.
    observed_data : SaltData
        Preprocessed supernova dataset (N SNe).
    mlim : tuple of float
        ``(min, max)`` apparent magnitude limits of the observable box.
    clim : tuple of float
        ``(min, max)`` colour limits.
    xlim : tuple of float
        ``(min, max)`` stretch limits.
    Nm, Nc, Nx : int
        Grid resolution along each axis.  Peak memory per SN chunk scales
        as W x ``sn_batch_size`` x Nm x Nc x Nx; reduce any of these if
        you encounter GPU OOM errors.
    sn_batch_size : int, optional
        Number of SNe processed per loop iteration.  Default is 32.
    device : torch.device or str, optional
        Computation device.  Default is ``'cuda'``.
    dtype : torch.dtype, optional
        Floating-point precision.  Default is ``torch.float32``, which is
        significantly faster on consumer GPUs.  Switch to ``torch.float64``
        if accuracy of the grid integral is a concern.
 
    Returns
    -------
    torch.Tensor, shape (W,)
        :math:`\\sum_n \\log S_n` for each walker, where :math:`S_n` is
        the Riemann-sum approximation of the selection probability integral
        for SN n.
 
    Notes
    -----
    Tensor shapes within each SN chunk (W = walkers, B = ``sn_batch_size``):
 
    - ``Sigma_inv``: ``(W, B, 3, 3)``
    - ``sE``: ``(W, B)``
    - ``y``, ``E_hat``, ``ll_grid``: ``(W, B, Nm, Nc, Nx)``
    - ``total`` accumulated across chunks: ``(W,)``
 
    See Also
    --------
    log_selection_probability_grid : Single-vector NumPy reference implementation.
    """
    W = global_params_batch.shape[0]
    N = observed_data.num_samples

    params = from_param_batch(
        torch.as_tensor(
            global_params_batch,
            dtype = dtype, device = device
        )
    )
    
    mean_int = get_mean_int_vectorized(
        **{p:params[p] for p in ['M0_int', 'alpha', 'beta_int',
                                 'c0_int', 'alphac_int', 'x0']}
    ) # (W, 3)
    cov_int = get_cov_int_vectorized(
        **{p:params[p] for p in ['alpha', 'beta_int', 'alphac_int',
                                 'sigma_int2', 'sigmac_int2', 'sigmax2']}
    ) # (W, 3, 3)

    dist_mod = torch.as_tensor(observed_data.dist_mod,
                               dtype = dtype, device = device)
    sigma_mu_z2 = torch.as_tensor(observed_data.sigma_mu_z2,
                                  dtype = dtype, device = device)
    cov = torch.as_tensor(observed_data.cov,
                          dtype = dtype, device = device)
    
    e1 = torch.as_tensor([1, 0, 0], dtype = dtype, device = device) # (3,)
    eE = torch.stack([params['RB'],
                      torch.ones(W, dtype = dtype, device = device),
                      torch.zeros(W, dtype = dtype, device = device)],
                      dim = 1) # (W, 3)
    e1e1 = torch.outer(e1, e1) # (3, 3)

    # grid: shared across walkers and SN chunks
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(mlim[0], mlim[1], Nm, dtype = dtype, device = device),
            torch.linspace(clim[0], clim[1], Nc, dtype =  dtype, device = device),
            torch.linspace(xlim[0], xlim[1], Nx, dtype = dtype, device = device),
            indexing = 'ij'
        ),
        dim = 0
    ) # (3, Nm, Nc, Nx)

    log_vol = torch.log(torch.as_tensor(
        (np.diff(mlim) * np.diff(clim) * np.diff(xlim))[0]
        / ((Nm - 1) * (Nc - 1) * (Nx - 1)),
        dtype = dtype, device = device
    ))

    total = torch.zeros(W, dtype = dtype, device = device)

    for start in range(0, N, sn_batch_size):
        end = min(start + sn_batch_size, N)
        B = end - start # batch length
        cov_b = cov[start:end] # (B, 3, 3)
        smz2_b = sigma_mu_z2[start:end] # (B,)
        dmod_b = dist_mod[start:end] # (B,)

        Sigma_inv = torch.linalg.inv( # (W, B, 3, 3)
            cov_int[:, None, :, :]
            + cov_b[None, :, :, :]
            + smz2_b[None, :, None, None] * e1e1[None, None, :, :]
        )

        sE = 1 / torch.sqrt(
            torch.einsum('wi,wbij,wj->wb', eE, Sigma_inv, eE)
        ) # (W, B)
        sE_exp = sE[:, :, None, None, None] # (W, B, 1, 1, 1)
        del sE

        # residual y = grid - mean_int - dist_mod * e1: (W, B, 3, Nm, Nc, Nx)
        dm = torch.zeros(W, B, 3, Nm, Nc, Nx, dtype = dtype, device = device)
        dm[:, :, 0] = dmod_b[None, :, None, None, None]
        y = (
            grid[None, None, :, :, :, :] # (1, 1, 3, Nm, Nc, Nx)
            - mean_int[:, None, :, None, None, None] # (W, 1, 3, 1, 1, 1)
            - dm # (W, B, 3, Nm, Nc, Nx)
        )
        del dm

        E_hat = sE_exp ** 2 * torch.einsum(
            'wi, wbij,wbjklm->wbklm', eE, Sigma_inv, y
        ) # (W, B, Nm, Nc, Nx)

        log_prefactor = torch.log(sE_exp * torch.sqrt(torch.tensor(2) * torch.pi))

        v = y.clone()
        v[:, :, 0] -= params['RB'][:, None, None, None, None] * E_hat
        v[:, :, 1] -= E_hat
        del y

        log_norm_factor = -0.5 * (
            torch.einsum('wbiklm,wbij,wbjklm->wbklm', v, Sigma_inv, v)
            + 3 * torch.log(torch.tensor(2) * torch.pi)
            - torch.linalg.slogdet(Sigma_inv)[1][:, :, None, None, None]
        ) # (W, B, Nm, Nc, Nx)
        del v

        log_exp_factor = (
            -torch.log(params['tau'][:, None, None, None, None])
            + 0.5 * (sE_exp / params['tau'][:, None, None, None, None]) ** 2
            - E_hat / params['tau'][:, None, None, None, None]
        )

        log_cdf_factor = torch.special.log_ndtr(
            E_hat / sE_exp - sE_exp / params['tau'][:, None, None, None, None]
        )
        del E_hat, sE_exp

        ll_grid = log_prefactor + log_norm_factor + log_exp_factor + log_cdf_factor
        del log_prefactor, log_norm_factor, log_exp_factor, log_cdf_factor

        # logsumexp over the flattened grid (Nm * Nc * Nx) -> (W, B), then sum over B -> (W,)
        total += (
            torch.logsumexp(ll_grid.reshape(W, B, -1), dim = 2) + log_vol
        ).sum(dim = 1)

    return total # (W, )