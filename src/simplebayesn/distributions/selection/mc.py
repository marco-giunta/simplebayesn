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

def get_kde_interpolant_grids(c_sel, z_sel, c_com, z_com,
                        nc = 1000, nz = 1000,
                        eps=1e-8):
    cmin, cmax = np.min(c_com), np.max(c_com)
    zmin, zmax = np.min(z_com), np.max(z_com)

    kde_sel = gaussian_kde(np.vstack([c_sel, z_sel]))
    kde_com = gaussian_kde(np.vstack([c_com, z_com]))

    def sel_prob_unnorm(c, z, eps=eps):
        cz = np.vstack([c, z])
        return kde_sel(cz) / (kde_com(cz) + eps)
    
    integral = dblquad(sel_prob_unnorm, zmin, zmax, cmin, cmax)[0]

    def sel_prob(c, z, eps=eps):
        return sel_prob_unnorm(c, z, eps) / integral
    
    c_vec = np.linspace(cmin, cmax, nc)
    z_vec = np.linspace(zmin, zmax, nz)

    c_grid, z_grid = np.meshgrid(c_vec, z_vec, indexing='ij')

    positions = np.vstack([c_grid.ravel(), z_grid.ravel()])
    sel_prob_grid = sel_prob(positions[0], positions[1]).T.reshape(c_grid.shape)
    
    c_vec = jnp.asarray(c_vec)
    z_vec = jnp.asarray(z_vec)
    sel_prob_grid = jnp.asarray(sel_prob_grid)

    return c_vec, z_vec, sel_prob_grid

@jax.jit
def interpolate_selection_2d(c, z, c_vec, z_vec, sel_prob_grid):
    nc = len(c_vec)
    nz = len(z_vec)
    
    c_min = c_vec[0]
    c_max = c_vec[-1]
    z_min = z_vec[0]
    z_max = z_vec[-1]
    
    outside = ((c < c_min) | (c > c_max) | (z < z_min) | (z > z_max))
    
    c_idx = (c - c_min) / (c_max - c_min) * (nc - 1)
    z_idx = (z - z_min) / (z_max - z_min) * (nz - 1)
    
    c_idx = jnp.clip(c_idx, 0, nc - 1)
    z_idx = jnp.clip(z_idx, 0, nz - 1)
    
    c_i0 = jnp.floor(c_idx).astype(int)
    z_i0 = jnp.floor(z_idx).astype(int)
    c_i1 = jnp.minimum(c_i0 + 1, nc - 1)
    z_i1 = jnp.minimum(z_i0 + 1, nz - 1)
    
    c_frac = c_idx - c_i0
    z_frac = z_idx - z_i0
    
    val_00 = sel_prob_grid[c_i0, z_i0]
    val_01 = sel_prob_grid[c_i0, z_i1]
    val_10 = sel_prob_grid[c_i1, z_i0]
    val_11 = sel_prob_grid[c_i1, z_i1]
    
    val = (val_00 * (1 - c_frac) * (1 - z_frac) +
           val_01 * (1 - c_frac) * z_frac +
           val_10 * c_frac * (1 - z_frac) + 
           val_11 * c_frac * z_frac)
    
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
                                     c_grid = None, z_grid = None, sel_prob_grid = None,
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
        p = interpolate_selection_2d(c_app_obs, observed_data_z[:, None],
                                     c_grid, z_grid, sel_prob_grid).mean(axis=1)

    
    return jnp.sum(jnp.log(p))

def log_selection_probability_mc_clf(global_params: dict, observed_data: SaltData,
                                     clf, num_sim_per_sample: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    shape_sim = (observed_data.num_samples, num_sim_per_sample)
    x = rng.normal(loc = global_params['x0'], scale = np.sqrt(global_params['sigmax2']), size = shape_sim)
    c_int = rng.normal(loc = global_params['c0_int'] + global_params['alphac_int'] * x,
                       scale = np.sqrt(global_params['sigmac_int2']), size = shape_sim)
    M_int = rng.normal(loc = global_params['M0_int'] + global_params['alpha'] * x + global_params['beta_int'] * c_int,
                       scale = np.sqrt(global_params['sigma_int2']), size = shape_sim)
    
    E = rng.exponential(scale = global_params['tau'], size = shape_sim)
    M_ext = M_int + global_params['RB'] * E
    c_app = c_int + E

    # condition on observed z dist + dm linearization, and on observed redshift error + chosen sigma_pec
    dist_mod = observed_data.dist_mod[:, None] + np.sqrt(observed_data.sigma_mu_z2)[:, None] * rng.normal(size = shape_sim)
    m_app = M_ext + dist_mod

    # condition on observed salt cov dist
    mcx = (
        np.stack([m_app, c_app, x], axis = -1) +
        np.einsum('nij,nsj->nsi', np.linalg.cholesky(observed_data.cov), rng.normal(size = (*shape_sim, 3)))
    )
    m_app_obs = mcx[..., 0]
    c_app_obs = mcx[..., 1]
    x_obs     = mcx[..., 2]
    
    p = clf.predict_proba( # hardcode sklearn dependence + selected class=1 here?
        np.stack([m_app_obs, c_app_obs], axis = -1).reshape(-1, 2) # (N_sam, N_sim, 2) -> (N_sam * N_sim, 2)
    )[:, 1].reshape(shape_sim).mean(axis = 1) # (N_sam, N_sim) -> (N_sam, )

    return np.sum(np.log(p/(1-p)))