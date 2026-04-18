from ...utils.data import SaltData
from ...utils.param_array import from_param_batch
import numpy as np
import torch

@torch.no_grad
def log_selection_prob_mc_vectorized(
        global_params_batch: torch.Tensor,
        observed_data: SaltData,
        clim: tuple[float, float],
        xlim: tuple[float, float],
        num_sim_per_sample: int,
        seed: int = 0,
        device: torch.device | str = 'cuda',
        dtype: torch.dtype | str = torch.float32
) -> torch.Tensor:
    gen = torch.Generator(device = device)
    gen.manual_seed(seed)
    
    W = global_params_batch.shape[0]
    N = observed_data.num_samples
    K = num_sim_per_sample
    params = from_param_batch(
        torch.as_tensor(
            global_params_batch,
            device = device, dtype = dtype
        )
    )

    obs_dist_mod = torch.as_tensor(observed_data.dist_mod,
                                   dtype = dtype, device = device)
    sigma_mu_z2 = torch.as_tensor(observed_data.sigma_mu_z2,
                                  dtype = dtype, device = device)
    chol_cov = torch.linalg.cholesky(
        torch.as_tensor(observed_data.cov,
                        dtype = dtype, device = device)
    )

    x = (
        params['x0'][:, None, None]
        + torch.sqrt(params['sigmax2'][:, None, None])
        * torch.randn(W, N, K, dtype = dtype, device = device,
                      generator = gen)
    ) # (W, N, K)

    c_int = (
        params['c0_int'][:, None, None]
        + params['alphac_int'][:, None, None] * x
        + torch.sqrt(params['sigmac_int2'][:, None, None])
        * torch.randn(W, N, K, dtype = dtype, device = device,
                      generator = gen)
    ) # (W, N, K)

    M_int = (
        params['M0_int'][:, None, None]
        + params['alpha'][:, None, None] * x
        + params['beta_int'][:, None, None] * c_int
        + torch.sqrt(params['sigma_int2'][:, None, None])
        * torch.randn(W, N, K, dtype = dtype, device = device,
                      generator = gen)
    ) # (W, N, K)

    E = torch.as_tensor(
        np.random.default_rng(seed = seed).\
            exponential(
                scale = params['tau'][:, None, None].cpu(),
                size = (W, N, K)
            )
    ).to(device = device, dtype = dtype) # (W, N, K)

    c_app = c_int + E
    m_app = (
        M_int
        + params['RB'][:, None, None] * E
        + obs_dist_mod[None, :, None]
        + torch.sqrt(sigma_mu_z2[None, :, None])
        * torch.randn(W, N, K, dtype = dtype, device = device,
                      generator = gen)
    ) # (W, N, K)

    mcx = (
        torch.stack([m_app, c_app, x], dim = -1)
        + torch.einsum(
            'nij,wnkj->wnki', chol_cov, torch.randn(
                W, N, K, 3, dtype = dtype, device = device,
                generator = gen
            )
        )
    )

    c_app_obs = mcx[..., 1] # (W, N, K)
    x_obs     = mcx[..., 2] # (W, N, K)

    selected = (
        (c_app_obs > clim[0]) & (c_app_obs < clim[1]) &
        (x_obs     > xlim[0]) & (x_obs     < xlim[1])
    ) # (W, N, K) bool

    log_p = torch.log(
        selected.float().mean(dim = 2)
    ) # (W, N)

    return log_p.sum(dim = 1) # (W,)
