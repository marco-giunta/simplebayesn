import numpy as np

def get_priors_params_uniform_priors():
    """
    Return the prior hyperparameter dictionary for a fully uniform Gibbs prior.

    Constructs the ``priors_params`` argument expected by
    :func:`~simplebayesn.samplers.gibbs.sample_global_params`.  All conjugate
    priors are set to their improper uniform limits:

    - ``tau``:  Improper positive uniform prior (``alpha_prior = -1``,
      ``beta_prior = 0``) for the inverse-gamma full conditional.
    - ``RB``:   Flat Gaussian prior (``mean_prior = None``,
      ``std_prior = None``), which makes the sampler draw directly from the
      likelihood.
    - ``x0``, ``c0_int`` / ``alphac_int``, ``M0_int`` / ``alpha`` /
      ``beta_int``:  Improper flat normal priors
      (``inv_cov_prior = None``, ``mean_prior`` set to zero vectors of the
      appropriate dimension) for the coefficient vectors.
    - ``sigmax2``, ``sigmac_int2``, ``sigma_int2``:  Improper
      inverse-gamma prior (``alpha_prior = -1``, ``beta_prior = 0``), equivalent
      to a positive uniform prior.

    Returns
    -------
    dict
        Nested dictionary with keys ``'tau'``, ``'RB'``, ``'x'``, ``'c'``,
        ``'M'``.  Each value is itself a dict of prior hyperparameters
        consumed by the corresponding Gibbs step sampler.

    See Also
    --------
    get_priors_params_uniform_priors_invgamma_sigmac_int2 :
        Variant with an inverse-gamma prior on ``sigmac_int2``.
    get_priors_params_uniform_invgamma :
        Variant with inverse-gamma priors on all variance parameters.
    """
    return {
        'tau':{'alpha_prior':-1, 'beta_prior':0},
        'RB': {'mean_prior':None, 'std_prior':None},
        'x':  {'alpha_prior':-1, 'beta_prior':0, 'inv_cov_prior':None, 'mean_prior':np.array([0])},
        'c':  {'alpha_prior':-1, 'beta_prior':0, 'inv_cov_prior':None, 'mean_prior':np.array([0, 0])},
        'M':  {'alpha_prior':-1, 'beta_prior':0, 'inv_cov_prior':None, 'mean_prior':np.array([0, 0, 0])},
    }

def get_priors_params_uniform_priors_invgamma_sigmac_int2(alpha: float = 0.003, beta: float = 0.003):
    """
    Return a Gibbs prior dictionary with an inverse-gamma prior on ``sigmac_int2``,
    a positive uniform improper prior on the other variance / scale parameters,
    and an improper uniform prior on the others.

    Identical to :func:`get_priors_params_uniform_priors` except that the
    ``alpha_prior`` and ``beta_prior`` for the intrinsic colour variance
    (``'c'`` block) are set to the provided ``alpha`` and ``beta`` values
    instead of the improper flat defaults.

    Parameters
    ----------
    alpha : float, optional
        Shape parameter for the inverse-gamma prior on ``sigmac_int2``.
        Default is 0.003.
    beta : float, optional
        Scale parameter for the inverse-gamma prior on ``sigmac_int2``.
        Default is 0.003.

    Returns
    -------
    dict
        Prior parameter dictionary in the same format as
        :func:`get_priors_params_uniform_priors`.

    See Also
    --------
    get_priors_params_uniform_invgamma :
        Applies inverse-gamma priors to *all* variance parameters.
    """
    prior_params_dict = get_priors_params_uniform_priors()
    prior_params_dict['c']['alpha_prior'] = alpha
    prior_params_dict['c']['beta_prior'] = beta
    return prior_params_dict

def get_priors_params_uniform_invgamma(alpha: float = 0.003, beta: float = 0.003):
    """
    Return a Gibbs prior dictionary with inverse-gamma priors on all variance
    and scale parameters, and improper uniform priors on all the others.

    Identical to :func:`get_priors_params_uniform_priors` except that
    ``alpha_prior`` and ``beta_prior`` are set to the provided ``alpha``
    and ``beta`` for all four variance / scale parameters:
    ``sigma_int2`` (``'M'`` block), ``sigmac_int2`` (``'c'`` block),
    ``sigmax2`` (``'x'`` block), and ``tau``.

    Parameters
    ----------
    alpha : float, optional
        Common shape parameter for all inverse-gamma priors.  Default is 0.003.
    beta : float, optional
        Common scale parameter for all inverse-gamma priors.  Default is 0.003.

    Returns
    -------
    dict
        Prior parameter dictionary in the same format as
        :func:`get_priors_params_uniform_priors`.
    """
    prior_params_dict = get_priors_params_uniform_priors()
    # sigma_int2
    prior_params_dict['M']['alpha_prior'] = alpha
    prior_params_dict['M']['beta_prior'] = beta
    # sigmac_int2
    prior_params_dict['c']['alpha_prior'] = alpha
    prior_params_dict['c']['beta_prior'] = beta
    # sigmax2
    prior_params_dict['x']['alpha_prior'] = alpha
    prior_params_dict['x']['beta_prior'] = beta
    # tau
    prior_params_dict['tau']['alpha_prior'] = alpha
    prior_params_dict['tau']['beta_prior'] = beta
    return prior_params_dict