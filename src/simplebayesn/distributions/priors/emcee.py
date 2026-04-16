import numpy as np
from scipy.stats import invgamma

def uniform_marginal_log_prior(global_params):
    """
    Improper uniform log-prior over all global hyperparameters.

    Returns ``-inf`` if any of the variance / scale parameters
    (``sigma_int2``, ``sigmac_int2``, ``sigmax2``, ``tau``) are
    non-positive; returns ``0`` otherwise.  All other parameters are
    unconstrained.

    This prior is appropriate for an emcee run where positivity is the only
    hard constraint and no additional regularisation is desired.

    Parameters
    ----------
    global_params : dict
        Dictionary of global hyperparameter values.

    Returns
    -------
    float
        ``0.0`` if all positivity constraints are satisfied, ``-inf``
        otherwise.

    See Also
    --------
    uniform_marginal_log_prior_invgamma_sigmac_int2 :
        Variant that places an inverse-gamma prior on ``sigmac_int2``.
    uniform_invgamma_marginal_log_prior :
        Variant that places inverse-gamma priors on all four variance/scale
        parameters.
    """
    return -np.inf if np.any(np.array([
        global_params['sigma_int2'],
        global_params['sigmac_int2'],
        global_params['sigmax2'],
        global_params['tau']
    ]) <= 0) else 0

def uniform_marginal_log_prior_invgamma_sigmac_int2(global_params, alpha: float = 0.003, beta: float = 0.003):
    """
    Log-prior that is uniform over all parameters except ``sigmac_int2``.

    Returns ``-inf`` if any positivity constraint is violated (see
    :func:`uniform_marginal_log_prior`).  Otherwise returns the
    inverse-gamma log-PDF evaluated at ``sigmac_int2`` with shape ``alpha``
    and scale ``beta``.  This is useful when the intrinsic colour variance
    is poorly constrained by data and needs regularisation toward small but
    non-zero values.

    Parameters
    ----------
    global_params : dict
        Dictionary of global hyperparameter values.
    alpha : float, optional
        Shape parameter of the inverse-gamma prior on ``sigmac_int2``.
        Default is 0.003.
    beta : float, optional
        Scale parameter of the inverse-gamma prior on ``sigmac_int2``.
        Default is 0.003.

    Returns
    -------
    float
        Log-prior value: ``-inf`` if any positivity constraint is violated,
        otherwise ``InvGamma(alpha, beta).logpdf(sigmac_int2)``.

    See Also
    --------
    uniform_invgamma_marginal_log_prior :
        Places inverse-gamma priors on *all* variance/scale parameters.
    """
    return -np.inf if np.any(np.array([
        global_params['sigma_int2'],
        global_params['sigmac_int2'],
        global_params['sigmax2'],
        global_params['tau']
    ]) <= 0) else invgamma.logpdf(global_params['sigmac_int2'], a=alpha, scale=beta)

def uniform_invgamma_marginal_log_prior(global_params, alpha: float = 0.003, beta: float = 0.003):
    """
    Log-prior that is uniform over location parameters and inverse-gamma
    over all four variance / scale parameters.

    Returns ``-inf`` if any positivity constraint is violated.  Otherwise
    returns the sum of inverse-gamma log-PDFs evaluated at ``sigma_int2``,
    ``sigmac_int2``, ``sigmax2``, and ``tau``, each with shape ``alpha``
    and scale ``beta``.

    The default ``alpha = beta = 0.003`` corresponds to a weakly
    informative prior that penalises extreme values while remaining
    broadly uninformative over the plausible range.

    Parameters
    ----------
    global_params : dict
        Dictionary of global hyperparameter values.
    alpha : float, optional
        Common shape parameter for all four inverse-gamma priors.
        Default is 0.003.
    beta : float, optional
        Common scale parameter for all four inverse-gamma priors.
        Default is 0.003.

    Returns
    -------
    float
        Sum of inverse-gamma log-PDFs for the four variance/scale
        parameters, or ``-inf`` if any positivity constraint is violated.
    """
    return -np.inf if np.any(np.array([
        global_params['sigma_int2'],
        global_params['sigmac_int2'],
        global_params['sigmax2'],
        global_params['tau']
    ]) <= 0) else np.sum((
        invgamma.logpdf(global_params['sigma_int2'], a=alpha, scale=beta),
        invgamma.logpdf(global_params['sigmac_int2'], a=alpha, scale=beta),
        invgamma.logpdf(global_params['sigmax2'], a=alpha, scale=beta),
        invgamma.logpdf(global_params['tau'], a=alpha, scale=beta),
    ))