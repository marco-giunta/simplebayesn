import numpy as np
from scipy.special import log_ndtr
from ..utils.intrinsic import get_mean_int, get_cov_int, get_mean_int_vectorized, get_cov_int_vectorized
from ..utils.data import SaltData
import torch
from ..utils.param_array import from_param_batch

def marginal_loglikelihood(global_params: dict,
                           observed_data: SaltData):
    """
    Compute the total marginal log-likelihood of the observed data under the
    Simple-BayeSN hierarchical model.

    The marginalization yields a closed-form expression (eq. 37 of Mandel et al. 2017)
    involving a Gaussian normalisation factor, an exponential prior factor,
    and a log-Normal-CDF term (computed stably via ``scipy.special.log_ndtr``).

    Parameters
    ----------
    global_params : dict
        Dictionary of global hyperparameter values.  All keys returned by
        :func:`~simplebayesn.utils.param_array.from_param_array` must be
        present.
    observed_data : SaltData
        Preprocessed supernova dataset as returned by
        :func:`~simplebayesn.utils.preprocessing.preprocess_data`.

    Returns
    -------
    float
        Sum of the per-SN marginal log-likelihoods, i.e.
        :math:`\\sum_n \\log p(\\phi_n \\mid \\theta)`.
    """
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
    
    E_hat = sE ** 2 * np.einsum('i,nij,nj->n', eE, Sigma_inv, y)

    log_prefactor = np.log(sE * np.sqrt(2 * np.pi))
    v = y - E_hat[:, np.newaxis] * eE
    log_norm_factor = -0.5 * (
        np.einsum('ni,nij,nj->n', v, Sigma_inv, v) +
        # np.log((2*np.pi)**3 / np.linalg.det(Sigma_inv))
        3 * np.log(2 * np.pi) - np.linalg.slogdet(Sigma_inv)[1]
    )
    log_exp_factor = (
        -np.log(global_params['tau'])
        + 0.5 * (sE / global_params['tau']) ** 2
        - E_hat / global_params['tau']
    )
    log_cdf_factor = log_ndtr(E_hat / sE - sE / global_params['tau'])
    result = log_prefactor + log_norm_factor + log_exp_factor + log_cdf_factor

    return result.sum()

@torch.no_grad
def marginal_loglikelihood_vectorized(
        global_params_batch: torch.Tensor,
        observed_data: SaltData,
        device: torch.device | str = 'cpu',
        dtype: torch.dtype | str = torch.float64
) -> torch.Tensor:
    """
    Compute the marginal log-likelihood for W parameter vectors simultaneously.
 
    Fully vectorized PyTorch version of :func:`marginal_loglikelihood` with
    no Python loop over walkers or SNe.  All W x N combinations are evaluated
    in a single operation, making this suitable for emcee's ``vectorize=True``
    mode where all walker proposals arrive as a single ``(W, 11)`` array.
 
    Unlike the selection correction functions, the marginal likelihood is
    run on CPU by default (``device='cpu'``).  The dominant operations are
    ``(W, N, 3, 3)`` matrix inversions - batches of small matrices - for
    which CPU BLAS is typically more efficient than GPU.
 
    Parameters
    ----------
    global_params_batch : torch.Tensor or array-like, shape (W, 11)
        W global hyperparameter vectors stacked row-wise, in canonical
        :data:`~simplebayesn.utils.param_array.PARAM_KEYS` order.
        Converted to a tensor of the requested ``dtype`` and ``device``
        internally via ``torch.as_tensor``, so numpy arrays are accepted
        directly.
    observed_data : SaltData
        Preprocessed supernova dataset (N SNe).
    device : torch.device or str, optional
        Computation device.  Default is ``'cpu'``; the likelihood is run
        on CPU by default because batched ``(W, N, 3, 3)`` small-matrix
        operations are typically faster there than on GPU.
    dtype : torch.dtype, optional
        Floating-point precision.  Default is ``torch.float64``, which is
        recommended for the matrix inversions and tail evaluations of the
        log-normal CDF.
 
    Returns
    -------
    torch.Tensor, shape (W,)
        :math:`\\sum_n \\log p(\\phi_n \\mid \\theta_w)` for each walker w,
        i.e. the total marginal log-likelihood summed over all N SNe.
 
    Notes
    -----
    All intermediate tensors are fully vectorized over both the walker
    dimension W and the SN dimension N simultaneously:
 
    - ``Sigma_inv``: ``(W, N, 3, 3)``
    - ``sE``, ``E_hat``: ``(W, N)``
    - All ``log_*`` factor terms: ``(W, N)``
    - Final sum over N: ``(W,)``
 
    See Also
    --------
    marginal_loglikelihood : Scalar (single-vector) NumPy equivalent.
    """
    global_params_batch = torch.as_tensor(global_params_batch,
                                          dtype = dtype,
                                          device = device)

    W = global_params_batch.shape[0]
    params = from_param_batch(global_params_batch)

    mean_int = get_mean_int_vectorized(
        **{p:params[p] for p in ['M0_int', 'alpha', 'beta_int',
                                 'c0_int', 'alphac_int', 'x0']}
    ) # (W, 3)
    cov_int = get_cov_int_vectorized(
        **{p:params[p] for p in ['alpha', 'beta_int', 'alphac_int',
                                 'sigma_int2', 'sigmac_int2', 'sigmax2']}
    ) # (W, 3, 3)

    cov_obs = torch.as_tensor(observed_data.cov, dtype = dtype, device = device) # (N, 3, 3)
    sigma_mu_z2 = torch.as_tensor(observed_data.sigma_mu_z2, dtype = dtype, device = device) # (N,)
    dist_mod = torch.as_tensor(observed_data.dist_mod, dtype = dtype, device = device) # (N,)
    d = torch.as_tensor(np.column_stack([
        observed_data.m_app,
        observed_data.c_app,
        observed_data.x
    ]), dtype = dtype, device = device) # (N, 3)

    e1 = torch.as_tensor([1, 0, 0], dtype = dtype, device = device) # (3,)
    eE = torch.stack([
        params['RB'],
        torch.ones(W, dtype = dtype, device = device),
        torch.zeros(W, dtype = dtype, device = device)
    ], dim = 1) # (W, 3)
    e1e1 = torch.outer(e1, e1) # (3, 3)

    Sigma_inv = torch.linalg.inv(
        cov_int[:, None, :, :] + # (W, 1, 3, 3)
        cov_obs[None, :, :, :] + # (1, N, 3, 3)
        sigma_mu_z2[None, :, None, None] * e1e1[None, None, :, :] # (1, N, 3, 3)
    ) # (W, N, 3, 3)

    sE = 1 / torch.sqrt(
        torch.einsum('wi,wnij,wj->wn', eE, Sigma_inv, eE)
    ) # (W, N)

    y = (
        d[None, :, :] # (1, N, 3)
        - mean_int[:, None, :] # (W, 1, 3)
        - dist_mod[None, :, None] * e1[None, None, :] # (1, N, 3)
    ) # (W, N, 3)

    E_hat = sE ** 2 * torch.einsum('wi,wnij,wnj->wn', eE, Sigma_inv, y) # (W, N)

    log_prefactor = torch.log(sE * torch.sqrt(torch.tensor(2) * torch.pi))

    v = y - E_hat[:, :, None] * eE[:, None, :] # (W, N, 3)
    log_norm_factor = -0.5 * (
        torch.einsum('wni,wnij,wnj->wn', v, Sigma_inv, v)
        + 3 * torch.log(torch.tensor(2) * torch.pi)
        - torch.linalg.slogdet(Sigma_inv)[1]
    )

    log_exp_factor = (
        -torch.log(params['tau'][:, None])
        + 0.5 * (sE / params['tau'][:, None]) ** 2
        - E_hat / params['tau'][:, None]
    )

    log_cdf_factor = torch.special.log_ndtr(E_hat / sE - sE / params['tau'][:, None])

    return (log_prefactor + log_norm_factor + log_exp_factor + log_cdf_factor).sum(dim = 1)