import jax
import jax.numpy as jnp
from functools import partial
from ...utils.data import SaltData
import numpy as np
from scipy.stats import gaussian_kde
# from scipy.integrate import tplquad

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

def get_kde_interpolant_grids(m_sel, c_sel, z_sel,
                              m_com, c_com, z_com,
                              nm = 100, nc = 100, nz = 100,
                              eps = 1e-8,
                              n_mc_norm = 100000):
    m_min, m_max = np.min(m_com), np.max(m_com)
    c_min, c_max = np.min(c_com), np.max(c_com)
    z_min, z_max = np.min(z_com), np.max(z_com)

    kde_sel = gaussian_kde(np.vstack([m_sel, c_sel, z_sel]))
    kde_com = gaussian_kde(np.vstack([m_com, c_com, z_com]))

    def sel_prob_unnnorm(m, c, z, eps=eps):
        mcz = np.vstack([m, c, z])
        return kde_sel(mcz) / (kde_com(mcz) + eps)
    
    # print('Computing triple integral normalization...')
    # integral = tplquad(sel_prob_unnnorm, z_min, z_max, c_min, c_max, m_min, m_max)[0]
    rng = np.random.default_rng(1234)
    m_mc = rng.uniform(m_min, m_max, n_mc_norm)
    c_mc = rng.uniform(c_min, c_max, n_mc_norm)
    z_mc = rng.uniform(z_min, z_max, n_mc_norm)
    
    mcz_mc = np.vstack([m_mc, c_mc, z_mc])
    ratio_mc = kde_sel(mcz_mc) / (kde_com(mcz_mc) + eps)
    
    volume = (m_max - m_min) * (c_max - c_min) * (z_max - z_min)
    integral = np.mean(ratio_mc) * volume
    # print('Done.')

    def sel_prob(m, c, z, eps=eps):
        return sel_prob_unnnorm(m, c, z, eps) / integral
    
    m_vec = np.linspace(m_min, m_max, nm)
    c_vec = np.linspace(c_min, c_max, nc)
    z_vec = np.linspace(z_min, z_max, nz)

    m_grid, c_grid, z_grid = np.meshgrid(m_vec, c_vec, z_vec, indexing='ij')

    positions = np.vstack([m_grid.ravel(), c_grid.ravel(), z_grid.ravel()])
    sel_prob_grid = sel_prob(positions[0], positions[1], positions[2]).T.reshape(m_grid.shape)
    
    m_vec = jnp.asarray(m_vec)
    c_vec = jnp.asarray(c_vec)
    z_vec = jnp.asarray(z_vec)
    sel_prob_grid = jnp.asarray(sel_prob_grid)

    return m_vec, c_vec, z_vec, sel_prob_grid

@jax.jit
def interpolate_selection_3d(m, c, z, m_vec, c_vec, z_vec, sel_prob_grid):
    nm = len(m_vec)
    nc = len(c_vec)
    nz = len(z_vec)
    
    m_min, m_max = m_vec[0], m_vec[-1]
    c_min, c_max = c_vec[0], c_vec[-1]
    z_min, z_max = z_vec[0], z_vec[-1]
    
    outside = ((m < m_min) | (m > m_max) |
               (c < c_min) | (c > c_max) |
               (z < z_min) | (z > z_max))
    
    # Convert to grid indices: (m-m0) / dm, with dm = (m1-m0)/(nm-1)
    m_idx = (m - m_min) * ((nm - 1) / (m_max - m_min))
    c_idx = (c - c_min) * ((nc - 1) / (c_max - c_min))
    z_idx = (z - z_min) * ((nz - 1) / (z_max - z_min))
    
    # Clip indices
    m_idx = jnp.clip(m_idx, 0, nm - 1)
    c_idx = jnp.clip(c_idx, 0, nc - 1)
    z_idx = jnp.clip(z_idx, 0, nz - 1)
    
    # Get surrounding indices
    m_i0 = jnp.floor(m_idx).astype(int)
    c_i0 = jnp.floor(c_idx).astype(int)
    z_i0 = jnp.floor(z_idx).astype(int)
    m_i1 = jnp.minimum(m_i0 + 1, nm - 1)
    c_i1 = jnp.minimum(c_i0 + 1, nc - 1)
    z_i1 = jnp.minimum(z_i0 + 1, nz - 1)
    
    # Get fractional parts = increments for linear interpolation
    m_frac = m_idx - m_i0
    c_frac = c_idx - c_i0
    z_frac = z_idx - z_i0
    
    # Trilinear interpolation (8 corners of cube)
    val_000 = sel_prob_grid[m_i0, c_i0, z_i0]
    val_001 = sel_prob_grid[m_i0, c_i0, z_i1]
    val_010 = sel_prob_grid[m_i0, c_i1, z_i0]
    val_011 = sel_prob_grid[m_i0, c_i1, z_i1]
    val_100 = sel_prob_grid[m_i1, c_i0, z_i0]
    val_101 = sel_prob_grid[m_i1, c_i0, z_i1]
    val_110 = sel_prob_grid[m_i1, c_i1, z_i0]
    val_111 = sel_prob_grid[m_i1, c_i1, z_i1]
    
    # Interpolate
    val = (val_000 * (1 - m_frac) * (1 - c_frac) * (1 - z_frac) +
           val_001 * (1 - m_frac) * (1 - c_frac) * z_frac +
           val_010 * (1 - m_frac) * c_frac * (1 - z_frac) +
           val_011 * (1 - m_frac) * c_frac * z_frac +
           val_100 * m_frac * (1 - c_frac) * (1 - z_frac) +
           val_101 * m_frac * (1 - c_frac) * z_frac +
           val_110 * m_frac * c_frac * (1 - z_frac) +
           val_111 * m_frac * c_frac * z_frac)
    
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
                                     dist_mod_sim = None, sigma_mu_z2_sim = None, cov_sim = None, z_sim = None,
                                     use_kde_selection: bool = False,
                                     m_grid = None, c_grid = None, z_grid = None, sel_prob_grid = None,
                                     seed=0):
    dm = observed_data_dist_mod if dist_mod_sim is None else dist_mod_sim
    s_mu_z2 = observed_data_sigma_mu_z2 if sigma_mu_z2_sim is None else sigma_mu_z2_sim
    cov = observed_data_cov if cov_sim is None else cov_sim
    z = observed_data_z if z_sim is None else z_sim
    shape_sim = (len(dm), num_sim_per_sample)

    key_x, key_c, key_M, key_E, key_dist_mod, key_noise = jax.random.split(jax.random.key(seed), 6)
    x = x0 + jnp.sqrt(sigmax2) * jax.random.normal(key_x, shape_sim)
    c_int = c0_int + alphac_int * x + jnp.sqrt(sigmac_int2) * jax.random.normal(key_c, shape_sim)
    M_int = M0_int + alpha * x + beta_int * c_int + sigma_int2 * jax.random.normal(key_M, shape_sim)

    E = tau * jax.random.exponential(key_E, shape_sim)
    M_ext = M_int + RB * E
    c_app = c_int + E

    dist_mod = dm[:, None] + jnp.sqrt(s_mu_z2)[:, None] * jax.random.normal(key_dist_mod, shape_sim)
    m_app = M_ext + dist_mod

    mcx = (
        jnp.stack([m_app, c_app, x], axis=-1) +
        jnp.einsum('nij,nsj->nsi', jnp.linalg.cholesky(cov), jax.random.normal(key_noise, (*shape_sim, 3)))
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
        p = interpolate_selection_3d(
            m_app_obs, c_app_obs, z[:, None],
            m_grid, c_grid, z_grid,
            sel_prob_grid
        ).mean(axis=1)

    
    return jnp.sum(jnp.log(p))