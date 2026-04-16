import numpy as np
import pandas as pd
from scipy.optimize import minimize, approx_fprime
from ..distributions.likelihood import marginal_loglikelihood
from ..utils.param_array import (
    PARAM_KEYS,
    IDX_POSITIVE_PARAMS,
    to_param_array,
    from_param_array
)
from ..utils.data import SaltData
from dataclasses import dataclass

@dataclass(frozen=True)
class MaximumLikelihoodResults:
    """
    Immutable container for maximum-likelihood estimation results.

    Stores the MLE point estimate, parameter uncertainties (from the Fisher
    information matrix), and metadata about the optimisation run.

    Parameters
    ----------
    global_params : dict
        Dictionary mapping parameter names to their MLE values.
    global_params_errors : dict
        Dictionary mapping parameter names to their estimated 1-sigma
        uncertainties (square roots of the diagonal of the inverse Hessian).
    info : dict
        Metadata about the optimisation, with keys ``'method'``,
        ``'num_iter'``, and ``'epsilon'``.

    Methods
    -------
    summary()
        Return a tidy ``pandas.DataFrame`` of MLE estimates and errors.
    """
    global_params: dict
    global_params_errors: dict
    info: dict

    def summary(self):
        """
        Return a summary table of MLE estimates and their uncertainties.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with columns ``'global parameter'``, ``'MLE'``,
            and ``'MLE error'``, one row per parameter in
            :data:`~simplebayesn.utils.param_array.PARAM_KEYS` order.
        """
        return pd.DataFrame({
            'global parameter':PARAM_KEYS,
            'MLE':[self.global_params[p] for p in PARAM_KEYS],
            'MLE error':[self.global_params_errors[p] for p in PARAM_KEYS]
        })

def get_default_maximum_likelihood_bounds(eps = 1e-6):
    """
    Return the default parameter bounds for the L-BFGS-B optimiser.

    Provides physically motivated box constraints on all 11 global
    hyperparameters.  Variance and scale parameters are bounded below by
    ``eps`` to enforce positivity; other parameters use broad but finite
    intervals.

    Parameters
    ----------
    eps : float, optional
        Small positive lower bound applied to ``sigma_int2``,
        ``sigmac_int2``, ``sigmax2``, and ``tau``.  Must be strictly
        positive.  Default is 1e-6.

    Returns
    -------
    dict
        Dictionary mapping each parameter name to a ``(lower, upper)``
        tuple, in the same format accepted by
        :func:`scipy.optimize.minimize` with ``method='L-BFGS-B'``.

    Raises
    ------
    ValueError
        If ``eps <= 0``.
    """
    if eps <= 0:
        raise ValueError(f'eps must be positive, got {eps} instead')
    return {
        'M0_int':(-25, -18),
        'alpha':(-2, 2),
        'beta_int':(-2, 2),
        'sigma_int2':(eps, 1),
        'c0_int':(-1, 1),
        'alphac_int':(-1, 1),
        'sigmac_int2':(eps, 1),
        'x0':(-1, 1),
        'sigmax2':(eps, 2),
        'tau':(eps, 1),
        'RB':(2, 5)
    }

def NLL(x: np.ndarray, observed_data: SaltData):
    """
    Negative marginal log-likelihood (objective function for minimisation).

    Converts the flat parameter array ``x`` to a named dictionary and
    returns the negated value of
    :func:`~simplebayesn.distributions.likelihood.marginal_loglikelihood`.

    Parameters
    ----------
    x : np.ndarray, shape (11,)
        Flat global hyperparameter array in canonical
        :data:`~simplebayesn.utils.param_array.PARAM_KEYS` order.
    observed_data : SaltData
        Preprocessed supernova dataset.

    Returns
    -------
    float
        :math:`-\\log p(\\text{data} \\mid \\theta)`.
    """
    return -marginal_loglikelihood(from_param_array(x), observed_data)

def compute_hessian(fun, x0, epsilon=1e-5, *args):
    """
    Numerically estimate the Hessian matrix of ``fun`` at ``x0``.

    Uses a forward finite-difference approximation of the Jacobian of the
    gradient: the (i, j) element of the Hessian is approximated by

    .. math::

        H_{ij} \\approx
        \\frac{\\nabla_j f(x_0 + \\epsilon e_i) - \\nabla_j f(x_0)}{\\epsilon}

    where :math:`e_i` is the i-th unit vector and :math:`\\nabla_j` denotes
    the j-th component of the gradient (itself computed by
    ``scipy.optimize.approx_fprime``).

    Used to estimate the Fisher information matrix at the MLE, from which
    parameter standard errors are extracted as :math:`\\sqrt{(H^{-1})_{ii}}`.

    Parameters
    ----------
    fun : callable
        Scalar-valued function to differentiate.
    x0 : np.ndarray, shape (n,)
        Point at which to evaluate the Hessian.
    epsilon : float, optional
        Step size for finite differences.  Default is 1e-5.
    *args
        Additional positional arguments forwarded to ``fun``.

    Returns
    -------
    np.ndarray, shape (n, n)
        Numerically estimated Hessian matrix.  Note: the result is not
        symmetrised; symmetry depends on the smoothness of ``fun`` and the
        choice of ``epsilon``.
    """
    n = len(x0)
    hessian = np.zeros((n, n))
    ei = np.zeros(n)

    for i in range(n):
        ei[i] = epsilon
        grad_i = approx_fprime(x0 + ei, fun, epsilon, *args)
        grad_i0 = approx_fprime(x0, fun, epsilon, *args)
        hessian[:, i] = (grad_i - grad_i0) / epsilon
        ei[i] = 0.0
    return hessian

def maximum_likelihood_solver(initial_conditions: dict, observed_data: SaltData,
                              bounds: dict = None, method: str = 'L-BFGS-B',
                              num_iter: int = 1000, epsilon: float = 1e-5,
                              print_message: bool = True):
    """
    Find the maximum-likelihood estimate of the global hyperparameters.

    Minimises the negative marginal log-likelihood
    :func:`NLL` using ``scipy.optimize.minimize`` with the specified bounded
    optimisation method, then estimates parameter uncertainties from the
    numerical Hessian at the solution.

    Parameters
    ----------
    initial_conditions : dict
        Starting point for the optimiser, as a named-parameter dictionary.
        Typically the output of
        :func:`~simplebayesn.utils.initialize.sample_initial_values_uniform`
        (``marginal=True``).
    observed_data : SaltData
        Preprocessed supernova dataset.
    bounds : dict or None, optional
        Parameter bounds in the same format as
        :func:`get_default_maximum_likelihood_bounds`.  If ``None``, the
        default bounds are used.
    method : str, optional
        Optimisation method passed to ``scipy.optimize.minimize``.
        Default is ``'L-BFGS-B'``.
    num_iter : int, optional
        Maximum number of optimiser iterations.  Default is 1000.
    epsilon : float, optional
        Finite-difference step size for the numerical Hessian computation.
        Default is 1e-5.
    print_message : bool, optional
        If ``True``, print the ``scipy.optimize.OptimizeResult.message``
        on completion.  Default is ``True``.

    Returns
    -------
    MaximumLikelihoodResults
        Frozen dataclass containing ``global_params`` (MLE point estimates),
        ``global_params_errors`` (1-sigma uncertainties), and ``info``
        (metadata).

    Notes
    -----
    Parameter errors are computed as :math:`\\sqrt{(H^{-1})_{ii}}` where
    :math:`H` is the numerical Hessian of the NLL at the MLE.  This is an
    asymptotic approximation that may be unreliable if the MLE is near a
    boundary or the likelihood surface is non-quadratic.
    """
    if bounds is None:
        bounds = get_default_maximum_likelihood_bounds()

    x0 = to_param_array(initial_conditions)
    result = minimize(
        NLL,
        x0,
        args=(observed_data),
        bounds=[bounds[p] for p in PARAM_KEYS],
        method=method,
        options={'maxiter':num_iter}
    )

    if print_message:
        print(result.message)
    ml_global_params = from_param_array(result.x)
    fisher_info_matrix = compute_hessian(NLL, result.x, epsilon, observed_data)
    errors = np.sqrt(np.diag(np.linalg.inv(fisher_info_matrix)))
    mle_global_params_errors = dict(zip(PARAM_KEYS, errors))

    return MaximumLikelihoodResults(
        global_params = ml_global_params,
        global_params_errors = mle_global_params_errors,
        info = {
            'method':method,
            'num_iter':num_iter,
            'epsilon':epsilon,
        }
    )