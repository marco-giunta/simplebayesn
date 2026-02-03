import numpy as np
from scipy.stats import truncnorm, norm, multivariate_normal, invgamma
from tqdm import trange
from ..utils.intrinsic import get_mean_int, get_cov_int
from ..utils.data import GibbsChainData, SaltData

def batch_sample_latent_params(m_app, c_app, x, dist_mod, E,
                               observed_data: SaltData,
                               global_params,
                               rng = None):
    N = observed_data.num_samples #len(m_app)
    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, int):
        rng = np.random.default_rng(rng)

    # sample E
    phi = np.column_stack([m_app, c_app, x])        # (N, 3)
    mu_vecs = np.column_stack([dist_mod, np.zeros(N), np.zeros(N)])
    eE = np.array([global_params['RB'], 1, 0])       # (3,)

    mean_int = get_mean_int(global_params).flatten() # (3,)
    inv_cov_int = np.linalg.inv(get_cov_int(global_params)) # (3,3)
    a = np.sqrt(eE @ inv_cov_int @ eE)                        # scalar
    b = ((phi - mean_int - mu_vecs) @ inv_cov_int @ eE)          # (N,)
    mE = (b - 1/global_params['tau']) / a**2
    sE = 1/a
    E_new = truncnorm.rvs((0-mE)/sE, np.inf, loc=mE, scale=sE, size=N, random_state=rng)

    # sample phi = (m_app, c_app, x)
    d = np.column_stack([
        observed_data.m_app,
        observed_data.c_app,
        observed_data.x
    ]) # (N,3)
    E_vecs = np.outer(E_new, eE)
    
    cov_phi_arr = np.linalg.inv(observed_data.inv_cov + inv_cov_int) # batched inverse (N,3,3)
    
    rhs = (
        np.einsum("nij,nj->ni", observed_data.inv_cov, d)
        + (inv_cov_int @ (mean_int + E_vecs + mu_vecs).T).T
    )
    mean_phi_arr = np.einsum("nij,nj->ni", cov_phi_arr, rhs)

    phi_new = mean_phi_arr + np.einsum(
        'nij,nj->ni',
        np.linalg.cholesky(cov_phi_arr), # batched Cholesky (N,3,3)
        rng.normal(size=(N,3))
    ) # mean + L*z for n samples

    m_app_new, c_app_new, x_new = phi_new.T

    # sample distmod
    e1 = np.array([1, 0, 0])
    sigma_mu_z2 = observed_data.sigma_mu_z2
    dist_mod_obs = observed_data.dist_mod
    std_dist_mod = 1 / np.sqrt(
        (e1 @ inv_cov_int @ e1) +
        1 / sigma_mu_z2
    )
    mean_dist_mod = (std_dist_mod ** 2) * (
        dist_mod_obs / sigma_mu_z2 + (
            (phi_new - mean_int - E_vecs) @ inv_cov_int @ e1
        )
    )
    dist_mod_new = norm.rvs(loc=mean_dist_mod, scale=std_dist_mod, size=N, random_state=rng)

    return {
        'm_app': m_app_new,
        'c_app': c_app_new,
        'x': x_new,
        'dist_mod': dist_mod_new,
        'E': E_new
    }

def sample_tau(E, alpha_prior: float = None, beta_prior: float = None, rng = None):
    if alpha_prior is None and beta_prior is None:
        alpha_prior, beta_prior = -1, 0
    alpha_post = alpha_prior + len(E)
    beta_post  = beta_prior  + sum(E)
    return invgamma.rvs(a = alpha_post, scale = beta_post, random_state = rng)

def sample_RB(m_app, c_app, x, E, dist_mod,
              global_params,
              mean_prior = None, std_prior = None,
              rng = None):
    mean_int = get_mean_int(global_params).flatten()
    inv_cov_int = np.linalg.inv(get_cov_int(global_params))
    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])

    phi = np.column_stack((m_app, c_app, x))
    std_lkl = 1 / np.sqrt(
        sum(E ** 2) *
        (e1 @ inv_cov_int @ e1)
    )
    mean_lkl = std_lkl ** 2 * np.einsum(
        's,i,ij,sj->',
        E, e1, inv_cov_int,
        phi - mean_int - dist_mod[:, np.newaxis] * e1 - E[:, np.newaxis] * e2
    )

    if std_prior is not None and mean_prior is not None:
        std_post = 1/np.sqrt(1/std_prior**2 + 1/std_lkl**2)
        mean_post = std_post**2 * (mean_prior/std_prior**2 + mean_lkl/std_lkl**2)
        return norm.rvs(loc = mean_post, scale = std_post, random_state = rng)
    else:
        return norm.rvs(loc = mean_lkl, scale = std_lkl, random_state = rng)
    
def sample_coef_vec(X, y, sigma2, mean_prior, inv_cov_prior, rng = None):
    if inv_cov_prior is None:
        inv_cov_prior = np.zeros((X.shape[1], X.shape[1]))
    
    cov_post = np.linalg.inv(inv_cov_prior + (X.T @ X)/sigma2)
    mean_post = cov_post @ (inv_cov_prior @ mean_prior + (X.T @ y)/sigma2)
    
    return multivariate_normal.rvs(mean = mean_post, cov = cov_post, random_state = rng)

def sample_var(X, y, coef_vec, alpha_prior, beta_prior, rng = None):
    alpha_post = alpha_prior + len(y) / 2
    v = y.reshape((-1, 1)) - X @ coef_vec.reshape((-1, 1))
    beta_post = beta_prior + 0.5 * v.T @ v
    return invgamma.rvs(a = alpha_post, scale = beta_post, random_state = rng)

def get_int_params_vals(m_app, c_app, x, E, dist_mod, RB):
    M_int = m_app - dist_mod - RB*E
    c_int = c_app - E
    return np.column_stack((M_int, c_int, x))

def sample_global_params(m_app, c_app, x, E, dist_mod,
                         global_params, priors_params, rng = None):
    tau = sample_tau(E, **priors_params['tau'], rng = rng)
    RB = sample_RB(m_app, c_app, x, E, dist_mod, global_params, **priors_params['RB'], rng = rng)

    N = len(x)
    X = get_int_params_vals(m_app, c_app, x, E, dist_mod, RB)
    
    X_x = np.ones((N, 1))
    y_x = X[:, 2]
    x0 = sample_coef_vec(X_x, y_x, global_params['sigmax2'],
                         priors_params['x']['mean_prior'],
                         priors_params['x']['inv_cov_prior'], rng)
    sigmax2 = sample_var(X_x, y_x, np.array([x0]),
                         priors_params['x']['alpha_prior'],
                         priors_params['x']['beta_prior'], rng)
    
    X_c = np.column_stack([np.ones(N), X[:, 2]])
    y_c = X[:, 1]
    a_c = sample_coef_vec(X_c, y_c, global_params['sigmac_int2'],
                          priors_params['c']['mean_prior'],
                          priors_params['c']['inv_cov_prior'], rng)
    c0_int, alphac_int = a_c
    sigmac_int2 = sample_var(X_c, y_c, a_c,
                             priors_params['c']['alpha_prior'],
                             priors_params['c']['beta_prior'], rng)
    
    X_M = np.column_stack([np.ones(N), X[:, 2], X[:, 1]])
    y_M = X[:, 0]

    a_M = sample_coef_vec(X_M, y_M, global_params['sigma_int2'],
                          priors_params['M']['mean_prior'],
                          priors_params['M']['inv_cov_prior'], rng)
    M0_int, alpha, beta_int = a_M
    sigma_int2 = sample_var(X_M, y_M, a_M,
                            priors_params['M']['alpha_prior'],
                            priors_params['M']['beta_prior'], rng)

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
    }

def gibbs_sampler(initial_values, priors_params,
                  observed_data: SaltData, num_iter,
                  seed: int = None):
    rng = np.random.default_rng(seed)
    num_iter += 1

    gibbs_chain = GibbsChainData(num_iter, observed_data.num_samples)
    lp_current_vals = initial_values['latent_params']
    gp_current_vals = initial_values['global_params']

    gibbs_chain[0] = {**lp_current_vals, **gp_current_vals}

    for t in trange(1, num_iter):
        lp_current_vals = batch_sample_latent_params(
            **lp_current_vals,
            observed_data = observed_data,
            global_params = gp_current_vals,
            rng = rng
        )

        gp_current_vals = sample_global_params(
            **lp_current_vals,
            global_params = gp_current_vals,
            priors_params = priors_params,
            rng = rng
        )

        gibbs_chain[t] = {**lp_current_vals, **gp_current_vals}

    return gibbs_chain
