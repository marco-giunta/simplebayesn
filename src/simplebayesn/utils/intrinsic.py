import numpy as np

def get_mean_int(global_params: dict):
    """
    Compute the mean of the intrinsic (M_int, c_int, x) population distribution in the form of a 3D Gaussian,
    instead of the conditional P(M_int | c_int, x)P(c_int | x)P(x) conditional form of the Simple-BayeSN model.
 
    Parameters
    ----------
    global_params : dict
        Dictionary of global hyperparameter values. Required keys:
        ``'M0_int'``, ``'c0_int'``, ``'x0'``, ``'alpha'``,
        ``'beta_int'``, ``'alphac_int'``.
 
    Returns
    -------
    np.ndarray, shape (3, 1)
        Mean vector [mu_M, mu_c, mu_x]^T of the joint intrinsic distribution,
        in the order (M_int, c_int, x).
    """
    mu = np.array([global_params['M0_int'], global_params['c0_int'], global_params['x0']]).reshape((3, 1))
    A = np.array([
        [0, global_params['beta_int'], global_params['alpha']],
        [0, 0, global_params['alphac_int']],
        [0, 0, 0]
    ])
    M = np.linalg.inv(np.eye(3) - A)
    return M @ mu

def get_cov_int(global_params: dict):
    """
    Compute the covariance matrix of the intrinsic (M_int, c_int, x) population distribution in the form of a 3D Gaussian,
    instead of the conditional P(M_int | c_int, x)P(c_int | x)P(x) conditional form of the Simple-BayeSN model.
 
    Parameters
    ----------
    global_params : dict
        Dictionary of global hyperparameter values. Required keys:
        ``'alpha'``, ``'beta_int'``, ``'alphac_int'``,
        ``'sigma_int2'``, ``'sigmac_int2'``, ``'sigmax2'``.
 
    Returns
    -------
    np.ndarray, shape (3, 3)
        Covariance matrix of the joint intrinsic distribution in the order
        (M_int, c_int, x).
 
    See Also
    --------
    get_cov_int_analytic : Closed-form version of this calculation.
    get_mean_int : Corresponding mean vector.
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

def get_mean_int_numeric(global_params: dict):
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
    return M @ mu

def get_cov_int_numeric(global_params: dict):
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

def get_cov_int_analytic(global_params: dict):
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

def get_mean_int_analytic(global_params):
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

    mean = np.zeros((3, 1))

    mean[0, 0] = M0_int + beta_int * c0_int + (alpha + beta_int * alphac_int) * x0
    mean[1, 0] = c0_int + alphac_int * x0
    mean[2, 0] = x0
    
    return mean