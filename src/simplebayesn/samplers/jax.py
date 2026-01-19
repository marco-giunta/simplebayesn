import jax
import jax.numpy as jnp
import numpy as np
from ..utils.intrinsic import get_mean_int, get_cov_int
from ..utils.data import SaltData, GibbsChainData, SaltDataCompact
from typing import NamedTuple
from functools import partial


def batch_sample_latent_params(key,
                               m_app, c_app, x, dist_mod, E,
                               observed_data: SaltDataCompact,
                               num_samples: int,
                               global_params):
    new_key, phi_key, E_key, dist_mod_key = jax.random.split(key, 4)
    N = num_samples # observed_data.num_samples
    phi = jnp.column_stack([m_app, c_app, x])
    mu_vecs = jnp.column_stack([dist_mod, jnp.zeros(N), jnp.zeros(N)])
    eE = jnp.asarray([global_params['RB'], 1, 0])

    mean_int = jnp.asarray(get_mean_int(global_params).flatten())
    inv_cov_int = jnp.linalg.inv(get_cov_int(global_params))
    a = jnp.sqrt(eE @ inv_cov_int @ eE)
    b = ((phi - mean_int - mu_vecs) @ inv_cov_int @ eE)
    mE = (b - 1/global_params['tau']) / a**2
    sE = 1/a
    E_new = mE + sE * jax.random.truncated_normal(E_key, (0-mE)/sE, jnp.inf, N)
    
    d = jnp.column_stack([
        observed_data.m_app,
        observed_data.c_app,
        observed_data.x
    ])
    E_vecs = jnp.outer(E_new, eE)

    cov_phi_arr = jnp.linalg.inv(observed_data.inv_cov + inv_cov_int)

    rhs = (
        jnp.einsum("nij,nj->ni", observed_data.inv_cov, d)
        + (inv_cov_int @ (mean_int + E_vecs + mu_vecs).T).T
    )
    mean_phi_arr = jnp.einsum("nij,nj->ni", cov_phi_arr, rhs)
    phi_new = mean_phi_arr + jnp.einsum(
        'nij,nj->ni',
        jnp.linalg.cholesky(cov_phi_arr),
        jax.random.normal(phi_key, (N, 3))
    )

    m_app_new, c_app_new, x_new = phi_new.T
    
    e1 = jnp.array([1, 0, 0])
    sigma_mu_z2 = jnp.asarray(observed_data.sigma_mu_z2)
    dist_mod_obs = jnp.asarray(observed_data.dist_mod)
    std_dist_mod = 1 / jnp.sqrt(
        (e1 @ inv_cov_int @ e1) +
        1 / sigma_mu_z2
    )
    mean_dist_mod = (std_dist_mod ** 2) * (
        dist_mod_obs / sigma_mu_z2 + (
            (phi_new - mean_int - E_vecs) @ inv_cov_int @ e1
        )
    )
    dist_mod_new = mean_dist_mod + std_dist_mod * jax.random.normal(dist_mod_key, N)

    return {
        'm_app': m_app_new,
        'c_app': c_app_new,
        'x': x_new,
        'dist_mod': dist_mod_new,
        'E': E_new
    }, new_key


def sample_tau(key, E, alpha_prior: float = None, beta_prior: float = None):
    new_key, tau_key = jax.random.split(key, 2)
    if alpha_prior is None and beta_prior is None:
        alpha_prior, beta_prior = -1, 0
    alpha_post = alpha_prior + len(E)
    beta_post  = beta_prior  + sum(E)
    tau_sample = beta_post / jax.random.gamma(tau_key, a = alpha_post) # 1/(gamma(alpha, 1)/beta)
    return tau_sample, new_key

def sample_RB(key,
              m_app, c_app, x, E, dist_mod,
              global_params,
              mean_prior = None, std_prior = None):
    new_key, RB_key = jax.random.split(key)
    
    mean_int = jnp.asarray(get_mean_int(global_params).flatten())
    inv_cov_int = jnp.linalg.inv(get_cov_int(global_params))
    e1 = jnp.array([1, 0, 0])
    e2 = jnp.array([0, 1, 0])

    phi = jnp.column_stack((m_app, c_app, x))
    std_lkl = 1 / jnp.sqrt(
        sum(E ** 2) *
        (e1 @ inv_cov_int @ e1)
    )
    mean_lkl = std_lkl ** 2 * jnp.einsum(
        's,i,ij,sj->',
        E, e1, inv_cov_int,
        phi - mean_int - dist_mod[:, jnp.newaxis] * e1 - E[:, jnp.newaxis] * e2
    )

    if std_prior is not None and mean_prior is not None:
        std_post = 1/jnp.sqrt(1/std_prior**2 + 1/std_lkl**2)
        mean_post = std_post**2 * (mean_prior/std_prior**2 + mean_lkl/std_lkl**2)
        RB_sample = mean_post + std_post * jax.random.normal(RB_key)
    else:
        RB_sample = mean_lkl + std_lkl * jax.random.normal(RB_key)
    
    return RB_sample, new_key

def sample_coef_vec(key, X, y, sigma2, mean_prior, inv_cov_prior):
    new_key, coef_vec_key = jax.random.split(key)
    if inv_cov_prior is None:
        inv_cov_prior = jnp.zeros((X.shape[1], X.shape[1]))
    
    cov_post = jnp.linalg.inv(inv_cov_prior + (X.T @ X)/sigma2)
    mean_post = cov_post @ (inv_cov_prior @ mean_prior + (X.T @ y)/sigma2)
    
    coef_vec_sample = jax.random.multivariate_normal(key = coef_vec_key, mean = mean_post, cov = cov_post)
    return coef_vec_sample, new_key

def sample_var(key, X, y, coef_vec, alpha_prior, beta_prior):
    new_key, var_key = jax.random.split(key)
    alpha_post = alpha_prior + len(y) / 2
    v = y.reshape((-1, 1)) - X @ coef_vec.reshape((-1, 1))
    beta_post = beta_prior + 0.5 * v.T @ v
    var_sample = beta_post / jax.random.gamma(var_key, alpha_post)
    return var_sample, new_key

def get_int_params_vals(m_app, c_app, x, E, dist_mod, RB):
    M_int = m_app - dist_mod - RB*E
    c_int = c_app - E
    return jnp.column_stack((M_int, c_int, x))

def sample_global_params(key,
                         m_app, c_app, x, E, dist_mod,
                         global_params, priors_params):
    new_key, tau_key, RB_key, x0_key, sigmax2_key, a_c_key, sigmac_int2_key, a_M_key, sigma_int2_key = jax.random.split(key, 9)
    tau = sample_tau(tau_key, E, **priors_params['tau'])
    RB = sample_RB(RB_key, m_app, c_app, x, E, dist_mod, global_params, **priors_params['RB'])
    
    N = len(x)
    X = get_int_params_vals(m_app, c_app, x, E, dist_mod, RB)
    
    X_x = jnp.ones((N, 1))
    y_x = X[:, 2]
    x0 = sample_coef_vec(x0_key, X_x, y_x, global_params['sigmax2'],
                         priors_params['x']['mean_prior'],
                         priors_params['x']['inv_cov_prior'])
    sigmax2 = sample_var(sigmax2_key, X_x, y_x, jnp.asarray([x0]),
                         priors_params['x']['alpha_prior'],
                         priors_params['x']['beta_prior'])
    
    X_c = jnp.column_stack([jnp.ones(N), X[:, 2]])
    y_c = X[:, 1]
    a_c = sample_coef_vec(a_c_key, X_c, y_c, global_params['sigmac_int2'],
                          priors_params['c']['mean_prior'],
                          priors_params['c']['inv_cov_prior'])
    c0_int, alphac_int = a_c
    sigmac_int2 = sample_var(sigmac_int2_key, X_c, y_c, a_c,
                             priors_params['c']['alpha_prior'],
                             priors_params['c']['beta_prior'])
    X_M = jnp.column_stack([jnp.ones(N), X[:, 2], X[:, 1]])
    y_M = X[:, 0]

    a_M = sample_coef_vec(a_M_key, X_M, y_M, global_params['sigma_int2'],
                          priors_params['M']['mean_prior'],
                          priors_params['M']['inv_cov_prior'])
    M0_int, alpha, beta_int = a_M
    sigma_int2 = sample_var(sigma_int2_key, X_M, y_M, a_M,
                            priors_params['M']['alpha_prior'],
                            priors_params['M']['beta_prior'])

    return {
        'tau':tau,
        'RB':RB,
        'x0':x0,
        'sigmax2':sigmax2,
        'c0_int':c0_int,
        'alphac_int':alphac_int,
        'sigmac_int2':sigmac_int2,
        'M0_int':M0_int,
        'alpha':alpha,
        'beta_int':beta_int,
        'sigma_int2':sigma_int2,
    }, new_key

def get_jax_priors(prior_dict):
    """
    Convert nested prior dicts into a JAX-compatible PyTree.

    Rules:
    - Keep None as None
    - Convert all numeric scalars to Python floats
    - Convert numpy arrays to jnp arrays
    - Recursively descend into dicts
    """
    def convert(x):
        if x is None:
            return None
        elif isinstance(x, (int, float)):
            return float(x)
        elif isinstance(x, (list, tuple)):
            return tuple(convert(xx) for xx in x)
        elif isinstance(x, dict):
            return {k: convert(v) for k, v in x.items()}
        else:
            # e.g. numpy arrays
            return jnp.asarray(x)

    return convert(prior_dict)

@partial(jax.jit, static_argnames = ("num_iter", "num_samples"))
def gibbs_sampler_single_chain(initial_values, priors_params,
                               observed_data: SaltDataCompact, num_iter,
                               num_samples: int,
                               seed: int = None):
    priors_params = get_jax_priors(priors_params)
    # observed_data = jax.tree_util.tree_map(
    #     lambda x: jnp.asarray(x) if isinstance(x, (np.ndarray, list)) else x,
    #     observed_data
    # )
    
    base_key = jax.random.PRNGKey(seed)
    keys = jax.random.split(base_key, num_iter)
    # num_iter += 1

    class GibbsState(NamedTuple):
        latent_params: dict
        global_params: dict

    init_state = GibbsState(
        latent_params = initial_values['latent_params'],
        global_params = initial_values['global_params']
    )

    def gibbs_step(state: GibbsState, key):
        latent_key, global_key = jax.random.split(key)

        latent_new, latent_key = batch_sample_latent_params(
            latent_key,
            **state.latent_params,
            observed_data=observed_data,
            global_params=state.global_params,
            num_samples=num_samples,
        )

        global_new, global_key = sample_global_params(
            global_key,
            **latent_new,
            global_params=state.global_params,
            priors_params=priors_params,
        )

        state_new = GibbsState(latent_new, global_new)
        return state_new, state_new

    final_state, all_states = jax.lax.scan(gibbs_step, init_state, keys)
    return final_state, all_states

def gibbs_sampler(initial_values_batch: tuple[dict], priors_params,
                  observed_data: SaltData, num_iter,
                  seed_batch: tuple[int] = None,
                  parallel: bool = False):
    assert len(initial_values_batch) == len(seed_batch)

    f = jax.pmap if parallel else jax.vmap
    batched_gibbs_sampler = f(
        gibbs_sampler_single_chain,
        in_axes = (0, None, None, None, 0), # initial_values, prior_params, observed_data, num_iter, seed
        axis_name = 'chains',
    )
    final_states, all_states = batched_gibbs_sampler(
        initial_values_batch,
        priors_params,
        observed_data,
        num_iter,
        seed_batch,
    )
    return final_states, all_states

def pytree_to_gibbs_chain(all_states, observed_data: SaltData):
    num_iter = all_states['global_params']['tau'].shape[0]
    num_data = observed_data.num_samples
    chain = GibbsChainData(num_iter, num_data)

    for name, arr in all_states['latent_params'].items():
        chain.latent_params[name][:] = np.array(arr)
    for name, arr in all_states['global_params'].items():
        chain.global_params[name][:] = np.array(arr)

    return chain