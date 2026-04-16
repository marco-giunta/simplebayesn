import numpy as np
from scipy.stats import truncnorm, norm, multivariate_normal, invgamma
from tqdm import trange
from ..utils.intrinsic import get_mean_int, get_cov_int
from ..utils.data import GibbsChainData, SaltData

def batch_sample_latent_params(m_app, c_app, x, dist_mod, E,
                               observed_data: SaltData,
                               global_params,
                               rng = None):
    """
    Draw a joint sample of all per-SN latent parameters (one full Gibbs step).

    Samples all N per-SN latent variables jointly from their conditional
    posterior given the current global parameters and observed data.  The
    sampling order is:

    1. **E** (dust reddening): truncated-normal full conditional, derived by
       projecting the joint (phi, E) Gaussian onto the E axis while enforcing
       E >= 0 via a half-space truncation.
    2. **phi = (m_app, c_app, x)**: multivariate-normal full conditional,
       combining the measurement-noise likelihood with the intrinsic Gaussian
       prior (shifted by E and the distance modulus).  A batched Cholesky
       decomposition is used for efficiency.
    3. **dist_mod**: normal full conditional, combining the cosmological
       distance modulus prior (centred on ``observed_data.dist_mod`` with
       variance ``sigma_mu_z2``) with the magnitude axis of the intrinsic
       likelihood.

    All N supernovae are sampled in a single vectorised call (no Python loop).

    Parameters
    ----------
    m_app, c_app, x, dist_mod, E : np.ndarray, shape (N,)
        Current latent-parameter values (from the previous Gibbs iteration).
    observed_data : SaltData
        Observed supernova dataset.
    global_params : dict
        Current global hyperparameter values.
    rng : np.random.Generator, int, or None, optional
        Random number generator.  If an integer, a new generator is seeded
        with that value.  If ``None``, a fresh default generator is created.

    Returns
    -------
    dict
        Dictionary with keys ``'m_app'``, ``'c_app'``, ``'x'``,
        ``'dist_mod'``, ``'E'``, each a 1-D ``np.ndarray`` of length N.
    """
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
    """
    Draw a sample of the dust scale parameter ``tau`` from its full conditional.

    The full conditional is an inverse-gamma distribution, conjugate to the
    exponential likelihood for E:

    .. math::

        \\tau \\mid E \\sim \\text{InvGamma}(\\alpha_0 + N,\\; \\beta_0 + \\sum_n E_n)

    The default ``alpha_prior = -1``, ``beta_prior = 0`` corresponds to the
    improper Jeffreys-style prior that makes the posterior proper as long as
    at least one :math:`E_n > 0`.

    Parameters
    ----------
    E : np.ndarray, shape (N,)
        Current dust reddening values for all N supernovae.
    alpha_prior : float or None, optional
        Shape hyperparameter of the inverse-gamma prior.  Default is ``-1``
        (improper uniform on ``tau``).
    beta_prior : float or None, optional
        Scale hyperparameter of the inverse-gamma prior.  Default is ``0``.
    rng : np.random.Generator or None, optional
        Random state passed to ``scipy.stats.invgamma.rvs``.

    Returns
    -------
    float
        A single sample of ``tau``.
    """
    if alpha_prior is None and beta_prior is None:
        alpha_prior, beta_prior = -1, 0
    alpha_post = alpha_prior + len(E)
    beta_post  = beta_prior  + sum(E)
    return invgamma.rvs(a = alpha_post, scale = beta_post, random_state = rng)

def sample_RB(m_app, c_app, x, E, dist_mod,
              global_params,
              mean_prior = None, std_prior = None,
              rng = None):
    """
    Draw a sample of the dust extinction ratio ``R_B`` from its full conditional.

    The full conditional for ``R_B`` given the latent intrinsic variables and
    dust values is Gaussian, derived by treating the magnitude axis of the
    intrinsic likelihood as a linear regression of
    :math:`M_{\\rm int} - (\\text{other terms})` on E with coefficient R_B.

    If ``mean_prior`` and ``std_prior`` are both provided, a Gaussian prior is
    combined with the likelihood in closed form (Gaussian-Gaussian conjugacy).
    Otherwise, the draw is taken directly from the likelihood (flat prior).

    Parameters
    ----------
    m_app, c_app, x : np.ndarray, shape (N,)
        Current latent photometric variables.
    E : np.ndarray, shape (N,)
        Current dust reddening values.
    dist_mod : np.ndarray, shape (N,)
        Current latent distance moduli.
    global_params : dict
        Current global hyperparameter values (used to evaluate the intrinsic
        mean and covariance at the current ``R_B``).
    mean_prior : float or None, optional
        Mean of the Gaussian prior on ``R_B``.  If ``None``, a flat prior is
        used. Default is ``None``.
    std_prior : float or None, optional
        Standard deviation of the Gaussian prior on ``R_B``.  If ``None``,
        a flat prior is used. Default is ``None``.
    rng : np.random.Generator or None, optional
        Random state.

    Returns
    -------
    float
        A single sample of ``R_B``.
    """
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
    """
    Draw a sample of a coefficient vector from a Gaussian-linear full conditional.

    Implements the standard Bayesian linear regression conjugate update:

    .. math::

        \\Sigma_{\\rm post} &= (\\Lambda_0 + X^\\top X / \\sigma^2)^{-1} \\\\
        \\mu_{\\rm post}    &= \\Sigma_{\\rm post}
                               (\\Lambda_0 \\mu_0 + X^\\top y / \\sigma^2)

    where :math:`\\Lambda_0` is ``inv_cov_prior`` and :math:`\\mu_0` is
    ``mean_prior``.  If ``inv_cov_prior`` is ``None``, a zero precision matrix
    is used (improper flat prior), in which case the posterior reduces to the
    OLS mean with variance :math:`\\sigma^2 (X^T X)^{-1}`.

    Used internally by :func:`sample_global_params` to sample ``x0``,
    ``(c0_int, alphac_int)``, and ``(M0_int, alpha, beta_int)`` from their
    respective linear-regression full conditionals.

    Parameters
    ----------
    X : np.ndarray, shape (N, p)
        Design matrix.
    y : np.ndarray, shape (N,)
        Response vector.
    sigma2 : float
        Current noise variance (the residual variance for this regression).
    mean_prior : np.ndarray, shape (p,)
        Prior mean vector.
    inv_cov_prior : np.ndarray, shape (p, p) or None
        Prior precision matrix.  If ``None``, a zero matrix is used.
    rng : np.random.Generator or None, optional
        Random state.

    Returns
    -------
    np.ndarray, shape (p,)
        A single draw from the posterior over coefficient vectors.
    """
    if inv_cov_prior is None:
        inv_cov_prior = np.zeros((X.shape[1], X.shape[1]))
    
    cov_post = np.linalg.inv(inv_cov_prior + (X.T @ X)/sigma2)
    mean_post = cov_post @ (inv_cov_prior @ mean_prior + (X.T @ y)/sigma2)
    
    return multivariate_normal.rvs(mean = mean_post, cov = cov_post, random_state = rng)

def sample_var(X, y, coef_vec, alpha_prior, beta_prior, rng = None):
    """
    Draw a sample of a residual variance from its inverse-gamma full conditional.

    Given the current coefficient vector and residuals, the conjugate posterior
    is:

    .. math::

        \\sigma^2 \\mid \\cdot \\sim
        \\text{InvGamma}\\!\\left(\\alpha_0 + N/2,\\;
                                  \\beta_0 + \\tfrac{1}{2}\\|y - X b\\|^2\\right)

    Used internally by :func:`sample_global_params` to sample ``sigmax2``,
    ``sigmac_int2``, and ``sigma_int2``.

    Parameters
    ----------
    X : np.ndarray, shape (N, p)
        Design matrix.
    y : np.ndarray, shape (N,)
        Response vector.
    coef_vec : np.ndarray, shape (p,)
        Current coefficient vector (used to compute residuals).
    alpha_prior : float
        Shape hyperparameter of the inverse-gamma prior.
    beta_prior : float
        Scale hyperparameter of the inverse-gamma prior.
    rng : np.random.Generator or None, optional
        Random state.

    Returns
    -------
    float
        A single sample of the variance parameter.
    """
    alpha_post = alpha_prior + len(y) / 2
    v = y.reshape((-1, 1)) - X @ coef_vec.reshape((-1, 1))
    beta_post = beta_prior + 0.5 * v.T @ v
    return invgamma.rvs(a = alpha_post, scale = beta_post, random_state = rng)

def get_int_params_vals(m_app, c_app, x, E, dist_mod, RB):
    """
    Compute the intrinsic parameter matrix (M_int, c_int, x) from latent values.

    Transforms the observed/latent photometric variables and dust values into
    the intrinsic frame by removing the dust and distance-modulus contributions:

    .. math::

        M_{\\rm int} &= m_{\\rm app} - \\mu - R_B E \\\\
        c_{\\rm int} &= c_{\\rm app} - E \\\\
        x            &= x \\text{ (unchanged)}

    This intrinsic matrix is used as the response and design matrix inputs for
    the linear-regression Gibbs steps in :func:`sample_global_params`.

    Parameters
    ----------
    m_app, c_app, x, E, dist_mod : np.ndarray, shape (N,)
        Current latent parameter values.
    RB : float
        Current dust extinction ratio.

    Returns
    -------
    np.ndarray, shape (N, 3)
        Column-stacked array ``[M_int, c_int, x]``.
    """
    M_int = m_app - dist_mod - RB*E
    c_int = c_app - E
    return np.column_stack((M_int, c_int, x))

def sample_global_params(m_app, c_app, x, E, dist_mod,
                         global_params, priors_params, rng = None):
    """
    Draw a joint sample of all global hyperparameters (one full Gibbs step).

    Samples all 11 global hyperparameters from their conjugate full conditionals,
    using the current latent variables as sufficient statistics.  The sampling
    order is:

    1. ``tau``           via :func:`sample_tau`.
    2. ``RB``            via :func:`sample_RB`.
    3. ``x0``, ``sigmax2``               (linear regression on x).
    4. ``c0_int``, ``alphac_int``, ``sigmac_int2``  (linear regression on c_int ~ x).
    5. ``M0_int``, ``alpha``, ``beta_int``, ``sigma_int2``
       (linear regression on M_int ~ x, c_int).

    Each regression step calls :func:`sample_coef_vec` and :func:`sample_var`
    sequentially, using the current variance as the regression noise level.

    Parameters
    ----------
    m_app, c_app, x, E, dist_mod : np.ndarray, shape (N,)
        Current latent parameter values.
    global_params : dict
        Current global hyperparameter values (provides the current noise
        variances used in regression steps).
    priors_params : dict
        Prior hyperparameter dictionary as returned by one of the
        ``get_priors_params_*`` functions in
        :mod:`simplebayesn.distributions.priors.gibbs`.
    rng : np.random.Generator or None, optional
        Random state.

    Returns
    -------
    dict
        Dictionary of sampled global hyperparameters with the same keys as
        ``global_params``.
    """
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
    """
    Run the blocked Gibbs sampler for the SimpleBayeSN hierarchical model.

    Alternates between a full latent-parameter step
    (:func:`batch_sample_latent_params`) and a full global-parameter step
    (:func:`sample_global_params`) for ``num_iter`` iterations, storing every
    sample (including the initial values at index 0) in a
    :class:`~simplebayesn.utils.data.GibbsChainData` object.

    The chain is initialised from ``initial_values`` at index 0 and then runs
    a ``trange`` loop from 1 to ``num_iter`` (inclusive), so the returned
    chain has ``num_iter + 1`` rows.

    Parameters
    ----------
    initial_values : dict
        Nested dictionary with keys ``'latent_params'`` and
        ``'global_params'``, each mapping parameter names to initial values.
        Typically produced by
        :func:`~simplebayesn.utils.initialize.sample_initial_values_uniform`.
    priors_params : dict
        Prior hyperparameter dictionary as returned by one of the
        ``get_priors_params_*`` functions in
        :mod:`simplebayesn.distributions.priors.gibbs`.
    observed_data : SaltData
        Preprocessed supernova dataset.
    num_iter : int
        Number of Gibbs iterations to run (excluding the initial state).
    seed : int or None, optional
        Random seed for reproducibility.  Default is ``None``.

    Returns
    -------
    GibbsChainData
        Chain object with ``num_iter + 1`` rows (index 0 is the initial
        state) for all global and latent parameters.

    Examples
    --------
    ::

        from simplebayesn.utils.initialize import sample_initial_values_uniform
        from simplebayesn.distributions.priors.gibbs import get_priors_params_uniform_priors
        from simplebayesn.samplers import gibbs_sampler

        iv = sample_initial_values_uniform(num_samples=len(data.z), seed=0)
        priors = get_priors_params_uniform_priors()
        chain = gibbs_sampler(iv, priors, data, num_iter=5000, seed=42)
    """
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
