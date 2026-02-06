import jax
import jax.numpy as jnp
from functools import partial
from ...utils.data import SaltData, SaltDataCompact

def preprocess_arguments_log_selection_probability_mc_jax(observed_data: SaltData | SaltDataCompact, global_params):
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

@partial(jax.jit, static_argnames=[
    #'observed_data_dist_mod', 'observed_data_sigma_mu_z2', 'observed_data_cov',
    'observed_data_num_samples',
    'clim', 'xlim',
    'num_sim_per_sample', 'seed',
    'selection_function'
])
def log_selection_probability_mc_jax(tau, RB,
                                     x0, sigmax2,
                                     c0_int, alphac_int, sigmac_int2,
                                     M0_int, alpha, beta_int, sigma_int2,
                                     clim, xlim,
                                     observed_data_dist_mod, observed_data_sigma_mu_z2, observed_data_cov,
                                     observed_data_num_samples, observed_data_z,
                                     num_sim_per_sample, selection_function = None,
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
    
    if selection_function is None:
        p = (
            (c_app_obs > clim[0]) &
            (c_app_obs < clim[1]) &
            (x_obs > xlim[0]) &
            (x_obs < xlim[1])
        ).mean(axis=1)
    else:
        p = selection_function(c_app_obs, observed_data_z[:, None]).mean(axis=1)

    
    return jnp.sum(jnp.log(p))