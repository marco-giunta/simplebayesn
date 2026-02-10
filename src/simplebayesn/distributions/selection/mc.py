import jax
import jax.numpy as jnp
from functools import partial
from ...utils.data import SaltData
import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import dblquad

def preprocess_arguments_log_selection_probability_mc_jax(observed_data: SaltData, global_params):
    return {
        'observed_data_dist_mod':jnp.asarray(observed_data.dist_mod),
        'observed_data_sigma_mu_z2':jnp.asarray(observed_data.sigma_mu_z2),
        'observed_data_cov':jnp.asarray(observed_data.cov),
        'observed_data_num_samples':observed_data.num_samples,
        'observed_data_z':jnp.asarray(observed_data.z),
        'tau':global_params['tau'],
        'RB':global_params['RB'],
        'x0':global_params['x0'],
        'sigmax2':global_params['sigmax2'],
        'c0_int':global_params['c0_int'],
        'alphac_int':global_params['alphac_int'],
        'sigmac_int2':global_params['sigmac_int2'],
        'M0_int':global_params['M0_int'],
        'alpha':global_params['alpha'],
        'beta_int':global_params['beta_int'],
        'sigma_int2':global_params['sigma_int2'],
    }

def get_kde_interpolant_grids(c_sel, m_sel, c_com, m_com,
                              nc = 1000, nm = 1000,
                              eps = 1e-8):
    c_min, c_max = np.min(c_com), np.max(c_com)
    m_min, m_max = np.min(m_com), np.max(m_com)

    kde_sel = gaussian_kde(np.vstack([c_sel, m_sel]))
    kde_com = gaussian_kde(np.vstack([c_com, m_com]))

    def sel_prob_unnnorm(c, m, eps=eps):
        cm = np.vstack([c, m])
        return kde_sel(cm) / (kde_com(cm) + eps)
    
    integral = dblquad(sel_prob_unnnorm, m_min, m_max, c_min, c_max)[0]

    def sel_prob(c, m, eps=eps):
        return sel_prob_unnnorm(c, m, eps) / integral
    
    c_vec = np.linspace(c_min, c_max, nc)
    m_vec = np.linspace(m_min, m_max, nm)

    c_grid, m_grid = np.meshgrid(c_vec, m_vec, indexing='ij')

    positions = np.vstack([c_grid.ravel(), m_grid.ravel()])
    sel_prob_grid = sel_prob(positions[0], positions[1]).T.reshape(c_grid.shape)
    
    c_vec = jnp.asarray(c_vec)
    m_vec = jnp.asarray(m_vec)
    sel_prob_grid = jnp.asarray(sel_prob_grid)

    return c_vec, m_vec, sel_prob_grid

@jax.jit
def interpolate_selection_2d(c, m, c_vec, m_vec, sel_prob_grid):
    nc = len(c_vec)
    nm = len(m_vec)
    
    c_min = c_vec[0]
    c_max = c_vec[-1]
    m_min = m_vec[0]
    m_max = m_vec[-1]
    
    outside = ((c < c_min) | (c > c_max) |
               (m < m_min) | (m > m_max))
    
    # Convert to grid indices: (c-c0) / dc, with dc = (c1-c0)/(nc-1)
    c_idx = (c - c_min) * ((nc - 1) / (c_max - c_min))
    m_idx = (m - m_min) * ((nm - 1) / (m_max - m_min))
    
    # Clip indices
    c_idx = jnp.clip(c_idx, 0, nc - 1)
    m_idx = jnp.clip(m_idx, 0, nm - 1)
    
    # Get surrounding indices
    c_i0 = jnp.floor(c_idx).astype(int)
    m_i0 = jnp.floor(m_idx).astype(int)
    c_i1 = jnp.minimum(c_i0 + 1, nc - 1)
    m_i1 = jnp.minimum(m_i0 + 1, nm - 1)
    
    # Get fractional parts = increments for linear interpolation
    c_frac = c_idx - c_i0
    m_frac = m_idx - m_i0
    
    # Bilinear interpolation (4 corners of square)
    val_00 = sel_prob_grid[c_i0, m_i0]
    val_01 = sel_prob_grid[c_i0, m_i1]
    val_10 = sel_prob_grid[c_i1, m_i0]
    val_11 = sel_prob_grid[c_i1, m_i1]
    
    # Interpolate
    val = (val_00 * (1 - c_frac) * (1 - m_frac) +
           val_01 * (1 - c_frac) * m_frac +
           val_10 * c_frac * (1 - m_frac) + 
           val_11 * c_frac * m_frac)
    
    return jnp.where(outside, 0.0, val)


@partial(jax.jit, static_argnames=[
    #'observed_data_dist_mod', 'observed_data_sigma_mu_z2', 'observed_data_cov',
    'observed_data_num_samples',
    'clim', 'xlim',
    'num_sim_per_sample', 'seed',
    'use_kde_selection'
])
def log_selection_probability_mc_jax(tau, RB,
                                     x0, sigmax2,
                                     c0_int, alphac_int, sigmac_int2,
                                     M0_int, alpha, beta_int, sigma_int2,
                                     clim, xlim,
                                     observed_data_dist_mod, observed_data_sigma_mu_z2, observed_data_cov,
                                     observed_data_num_samples, observed_data_z,
                                     num_sim_per_sample,
                                     use_kde_selection: bool = False,
                                     c_grid = None, m_grid = None, sel_prob_grid = None,
                                     seed=0):
    
    key_x, key_c, key_M, key_E, key_dist_mod, key_noise = jax.random.split(jax.random.key(seed), 6)
    shape_sim = (observed_data_num_samples, num_sim_per_sample)
    x = x0 + jnp.sqrt(sigmax2) * jax.random.normal(key_x, shape_sim)
    c_int = c0_int + alphac_int * x + jnp.sqrt(sigmac_int2) * jax.random.normal(key_c, shape_sim)
    M_int = M0_int + alpha * x + beta_int * c_int + sigma_int2 * jax.random.normal(key_M, shape_sim)

    E = tau * jax.random.exponential(key_E, shape_sim)
    M_ext = M_int + RB * E
    c_app = c_int + E

    dist_mod = observed_data_dist_mod[:, None] + jnp.sqrt(observed_data_sigma_mu_z2)[:, None] * jax.random.normal(key_dist_mod, shape_sim)
    m_app = M_ext + dist_mod

    mcx = (
        jnp.stack([m_app, c_app, x], axis=-1) +
        jnp.einsum('nij,nsj->nsi', jnp.linalg.cholesky(observed_data_cov), jax.random.normal(key_noise, (*shape_sim, 3)))
    )
    m_app_obs = mcx[..., 0]
    c_app_obs = mcx[..., 1]
    x_obs = mcx[..., 2]
    
    if not use_kde_selection:
        p = (
            (c_app_obs > clim[0]) &
            (c_app_obs < clim[1]) &
            (x_obs > xlim[0]) &
            (x_obs < xlim[1])
        ).mean(axis=1)
    else:
        p = interpolate_selection_2d(c_app_obs, m_app_obs,
                                     c_grid, m_grid, sel_prob_grid).mean(axis=1)

    
    return jnp.sum(jnp.log(p))