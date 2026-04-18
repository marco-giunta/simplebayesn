import numpy as np
import torch

def get_mean_int(global_params: dict[str, float | np.floating], kind: str = 'analytic') -> np.ndarray:
    """
    Compute the mean of the intrinsic (M_int, c_int, x) population distribution in the form of a 3D Gaussian,
    instead of the conditional P(M_int | c_int, x)P(c_int | x)P(x) conditional form of the Simple-BayeSN model.

    This is a convenience wrapper to either get_mean_int_analytic or get_mean_int_numeric.
 
    Parameters
    ----------
    global_params : dict
        Dictionary of global hyperparameter values. Required keys:
        ``'M0_int'``, ``'c0_int'``, ``'x0'``, ``'alpha'``,
        ``'beta_int'``, ``'alphac_int'``.
    kind : str, default 'analytic'
        Kind of computation performed. If "analytic", the analytic expression
        for the intrinsic covariance is used; if "numeric", the numeric version
        based on matrix inversion is used. Any other value results in error.
 
    Returns
    -------
    np.ndarray, shape (3, 1)
        Mean vector [mu_M, mu_c, mu_x]^T of the joint intrinsic distribution,
        in the order (M_int, c_int, x).

    See also
    --------
    get_mean_int_analytic : Analytic version of this calculation.
    get_mean_int_numeric : Numeric version of this calculation.
    """
    if kind == 'analytic':
        return get_mean_int_analytic(global_params=global_params)
    elif kind == 'numeric':
        return get_mean_int_numeric(global_params=global_params)
    else:
        raise ValueError(f'Invalid {kind=}, choose "numeric" or "analytic" instead')

def get_cov_int(global_params: dict[str, float | np.floating], kind: str = 'analytic') -> np.ndarray:
    """
    Compute the covariance matrix of the intrinsic (M_int, c_int, x) population distribution in the form of a 3D Gaussian,
    instead of the conditional P(M_int | c_int, x)P(c_int | x)P(x) conditional form of the Simple-BayeSN model.

    This is a convenience wrapper to either get_cov_int_analytic or get_cov_int_numeric.
 
    Parameters
    ----------
    global_params : dict
        Dictionary of global hyperparameter values. Required keys:
        ``'alpha'``, ``'beta_int'``, ``'alphac_int'``,
        ``'sigma_int2'``, ``'sigmac_int2'``, ``'sigmax2'``.
    kind : str
        Kind of computation performed. If "analytic", the analytic expression
        for the intrinsic covariance is used; if "numeric", the numeric version
        based on matrix inversion is used. Any other value results in error.
 
    Returns
    -------
    np.ndarray, shape (3, 3)
        Covariance matrix of the joint intrinsic distribution in the order
        (M_int, c_int, x).
 
    See Also
    --------
    get_cov_int_analytic : Analytic version of this calculation.
    get_cov_int_numeric : Numeric version of this calculation.
    get_mean_int : Corresponding mean vector.
    """
    if kind == 'analytic':
        return get_cov_int_analytic(global_params=global_params)
    elif kind == 'numeric':
        return get_cov_int_numeric(global_params=global_params)
    else:
        raise ValueError(f'Invalid {kind=}, choose "numeric" or "analytic" instead')

def get_mean_int_numeric(global_params: dict) -> np.ndarray:
    """
    Numerically compute the mean of the intrinsic population distribution.
 
    Parameters
    ----------
    global_params : dict
        Dictionary of global hyperparameter values. Required keys:
        ``'M0_int'``, ``'c0_int'``, ``'x0'``, ``'alpha'``,
        ``'beta_int'``, ``'alphac_int'``.
 
    Returns
    -------
    np.ndarray, shape (3, 1)
        Mean vector [mu_M, mu_c, mu_x]^T of the joint intrinsic distribution.
 
    See Also
    --------
    get_mean_int_analytic : Closed-form equivalent.
    """
    mu = np.array([global_params['M0_int'], global_params['c0_int'], global_params['x0']]).reshape((3, 1))
    A = np.array([
        [0, global_params['beta_int'], global_params['alpha']],
        [0, 0, global_params['alphac_int']],
        [0, 0, 0]
    ])
    M = np.linalg.inv(np.eye(3) - A)
    return (M @ mu).flatten()

def get_cov_int_numeric(global_params: dict) -> np.ndarray:
    """
    Numerically compute the covariance of the intrinsic population distribution.
 
    Parameters
    ----------
    global_params : dict
        Dictionary of global hyperparameter values. Required keys:
        ``'alpha'``, ``'beta_int'``, ``'alphac_int'``,
        ``'sigma_int2'``, ``'sigmac_int2'``, ``'sigmax2'``.
 
    Returns
    -------
    np.ndarray, shape (3, 3)
        Covariance matrix of the joint intrinsic distribution.
 
    See Also
    --------
    get_cov_int_analytic : Closed-form equivalent.
    """
    A = np.array([
        [0, global_params['beta_int'], global_params['alpha']],
        [0, 0, global_params['alphac_int']],
        [0, 0, 0]
    ])
    M = np.linalg.inv(np.eye(3) - A)
    S = np.diag(np.array([global_params['sigma_int2'], global_params['sigmac_int2'], global_params['sigmax2']]))
    C = M @ S @ M.T
    return C

def get_cov_int_analytic(global_params: dict) -> np.ndarray:
    """
    Analytically compute the covariance of the intrinsic population distribution.
 
    Provides a closed-form evaluation of ``M @ S @ M^T`` (see
    :func:`get_cov_int_numeric`) by directly substituting the matrix product.
 
    The analytic expressions for each entry are:
 
    - ``C[0,0]`` = ``(alpha + beta_int * alphac_int)^2 * sigmax2 + beta_int^2 * sigmac_int2 + sigma_int2``
    - ``C[1,1]`` = ``alphac_int^2 * sigmax2 + sigmac_int2``
    - ``C[2,2]`` = ``sigmax2``
    - ``C[0,1]`` = ``(alpha + beta_int * alphac_int) * alphac_int * sigmax2 + beta_int * sigmac_int2``
    - ``C[0,2]`` = ``(alpha + beta_int * alphac_int) * sigmax2``
    - ``C[1,2]`` = ``alphac_int * sigmax2``
 
    Parameters
    ----------
    global_params : dict
        Dictionary of global hyperparameter values. Required keys:
        ``'alpha'``, ``'beta_int'``, ``'alphac_int'``,
        ``'sigma_int2'``, ``'sigmac_int2'``, ``'sigmax2'``.
 
    Returns
    -------
    np.ndarray, shape (3, 3)
        Symmetric covariance matrix of the joint intrinsic (M_int, c_int, x)
        distribution.
 
    See Also
    --------
    get_cov_int_numeric : Matrix-inversion-based equivalent.
    """
    alpha        = global_params['alpha']
    beta_int     = global_params['beta_int']
    alphac_int   = global_params['alphac_int']
    sigma_int2   = global_params['sigma_int2']
    sigmac_int2  = global_params['sigmac_int2']
    sigmax2      = global_params['sigmax2']

    C = np.zeros((3, 3))

    C[0, 0] = ((alpha + beta_int * alphac_int) ** 2) * sigmax2 \
               + (beta_int ** 2) * sigmac_int2 \
               + sigma_int2

    C[1, 1] = (alphac_int ** 2) * sigmax2 + sigmac_int2
    C[2, 2] = sigmax2

    C[0, 1] = (alpha + beta_int * alphac_int) * alphac_int * sigmax2 + beta_int * sigmac_int2
    C[1, 0] = C[0, 1]

    C[0, 2] = (alpha + beta_int * alphac_int) * sigmax2
    C[2, 0] = C[0, 2]

    C[1, 2] = alphac_int * sigmax2
    C[2, 1] = C[1, 2]

    return C

def get_mean_int_analytic(global_params: dict) -> np.ndarray:
    """
    Analytically compute the mean of the intrinsic population distribution.
 
    Provides a closed-form evaluation of ``(I - A)^{-1} @ mu`` (see
    :func:`get_mean_int_numeric`) by direct substitution. The reduced-form
    mean entries are:
 
    - ``mean[0]`` = ``M0_int + beta_int * c0_int + (alpha + beta_int * alphac_int) * x0``
    - ``mean[1]`` = ``c0_int + alphac_int * x0``
    - ``mean[2]`` = ``x0``
 
    Parameters
    ----------
    global_params : dict
        Dictionary of global hyperparameter values. Required keys:
        ``'M0_int'``, ``'c0_int'``, ``'x0'``, ``'alpha'``,
        ``'beta_int'``, ``'alphac_int'``.
 
    Returns
    -------
    np.ndarray, shape (3, 1)
        Mean vector [mu_M, mu_c, mu_x]^T of the joint intrinsic distribution.
 
    See Also
    --------
    get_mean_int_numeric : Matrix-inversion-based equivalent.
    """
    M0_int     = global_params['M0_int']
    c0_int     = global_params['c0_int']
    x0         = global_params['x0']
    alpha      = global_params['alpha']
    beta_int   = global_params['beta_int']
    alphac_int = global_params['alphac_int']

    mean = np.zeros(3)

    mean[0] = M0_int + beta_int * c0_int + (alpha + beta_int * alphac_int) * x0
    mean[1] = c0_int + alphac_int * x0
    mean[2] = x0
    
    return mean

def get_cov_int_vectorized(
        alpha: torch.Tensor | np.ndarray,
        beta_int: torch.Tensor | np.ndarray,
        alphac_int: torch.Tensor | np.ndarray,
        sigma_int2: torch.Tensor | np.ndarray,
        sigmac_int2: torch.Tensor | np.ndarray,
        sigmax2: torch.Tensor | np.ndarray
) -> torch.Tensor | np.ndarray:
    """
    Analytically compute the intrinsic covariance matrix for W parameter
    vectors simultaneously.

    Fully vectorized version of
    :func:`~simplebayesn.utils.intrinsic.get_cov_int_analytic`: no Python
    loop over walkers, all W matrices are built in a single tensor operation.

    This function is intended to be used alongside `emcee.EnsembleSampler` 
    with `vectorize=True`.

    Parameters
    ----------
    alpha, beta_int, alphac_int, sigma_int2, sigmac_int2, sigmax2 :
        torch.Tensor or np.ndarray, shape (W,)
        Per-walker hyperparameter values.

    Returns
    -------
    torch.Tensor or np.ndarray, shape (W, 3, 3)
        Intrinsic covariance matrices, one per walker.
    """
    if isinstance(alpha, torch.Tensor):
        C = torch.zeros((alpha.shape[0], 3, 3), dtype = alpha.dtype, device = alpha.device)
    else:
        C = np.zeros((alpha.shape[0], 3, 3))
        
    abc = alpha + beta_int * alphac_int

    C[:, 0, 0] = abc ** 2 * sigmax2 + beta_int ** 2 * sigmac_int2 + sigma_int2
    C[:, 1, 1] = alphac_int ** 2 * sigmax2 + sigmac_int2
    C[:, 2, 2] = sigmax2

    C[:, 0, 1] = abc * alphac_int * sigmax2 + beta_int * sigmac_int2
    C[:, 1, 0] = C[:, 0, 1]

    C[:, 0, 2] = abc * sigmax2
    C[:, 2, 0] = C[:, 0, 2]

    C[:, 1, 2] = alphac_int * sigmax2
    C[:, 2, 1] = C[:, 1, 2]

    return C

def get_mean_int_vectorized(
        M0_int: torch.Tensor | np.ndarray,
        alpha: torch.Tensor | np.ndarray,
        beta_int: torch.Tensor | np.ndarray,
        c0_int: torch.Tensor | np.ndarray,
        alphac_int: torch.Tensor | np.ndarray,
        x0: torch.Tensor | np.ndarray
) -> torch.Tensor | np.ndarray:
    """
    Compute the intrinsic population mean vector for W parameter vectors
    simultaneously.
 
    Fully vectorised version of :func:`get_mean_int` (analytic form) that
    operates on arrays of shape ``(W,)`` rather than a single scalar dict,
    returning all W mean vectors in one operation with no Python loop.
 
    Supports both PyTorch tensors and NumPy arrays; the return type matches
    the input type.
 
    The reduced-form mean entries are:
 
    .. math::
 
        \\mu_0 &= M_0^{\\rm int} + \\beta_{\\rm int}\\, c_0^{\\rm int}
                  + (\\alpha + \\beta_{\\rm int}\\, \\alpha_c^{\\rm int})\\, x_0 \\\\
        \\mu_1 &= c_0^{\\rm int} + \\alpha_c^{\\rm int}\\, x_0 \\\\
        \\mu_2 &= x_0
 
    Parameters
    ----------
    M0_int, alpha, beta_int, c0_int, alphac_int, x0 :
        torch.Tensor or np.ndarray, shape (W,)
        Per-walker hyperparameter values.
 
    Returns
    -------
    torch.Tensor or np.ndarray, shape (W, 3)
        Intrinsic mean vectors ``[mu_M, mu_c, mu_x]`` for each of the W
        walkers.
 
    See Also
    --------
    get_mean_int : Single-vector version accepting a parameter dict.
    get_cov_int_vectorized : Corresponding vectorised covariance function.
    """
    if isinstance(alpha, torch.Tensor):
        mean = torch.zeros((alpha.shape[0], 3), dtype = alpha.dtype, device = alpha.device)
    else:
        mean = np.zeros((alpha.shape[0], 3))

    mean[:, 0] = M0_int + beta_int * c0_int + (alpha + beta_int * alphac_int) * x0
    mean[:, 1] = c0_int + alphac_int * x0
    mean[:, 2] = x0

    return mean