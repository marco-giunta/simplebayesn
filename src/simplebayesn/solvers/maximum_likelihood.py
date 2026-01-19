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
from ..utils.data import SaltData, SaltDataCompact
from dataclasses import dataclass

@dataclass(frozen=True)
class MaximumLikelihoodResults:
    global_params: dict
    global_params_errors: dict
    info: dict

    def summary(self):
        return pd.DataFrame({
            'global parameter':PARAM_KEYS,
            'MLE':[self.global_params[p] for p in PARAM_KEYS],
            'MLE error':[self.global_params_errors[p] for p in PARAM_KEYS]
        })

def get_default_maximum_likelihood_bounds(eps = 1e-6):
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

def NLL(x: np.ndarray, observed_data: SaltData | SaltDataCompact):
    return -marginal_loglikelihood(from_param_array(x), observed_data)

def compute_hessian(fun, x0, epsilon=1e-5, *args):
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

def maximum_likelihood_solver(initial_conditions: dict, observed_data: SaltData | SaltDataCompact,
                              bounds: dict = None, method: str = 'L-BFGS-B',
                              num_iter: int = 1000, epsilon: float = 1e-5,
                              print_message: bool = True):
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