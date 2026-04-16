import numpy as np

PARAM_KEYS = [
    'M0_int', 'alpha', 'beta_int', 'sigma_int2',
    'c0_int', 'alphac_int', 'sigmac_int2', 'x0',
    'sigmax2', 'tau', 'RB'
]
"""
Ordered list of global hyperparameter names.

This ordering defines the canonical mapping between dictionary-style parameter
representations and flat 1-D arrays, as used by the emcee sampler interface.
The 11 parameters are, in order:

``M0_int``, ``alpha``, ``beta_int``, ``sigma_int2``,
``c0_int``, ``alphac_int``, ``sigmac_int2``, ``x0``,
``sigmax2``, ``tau``, ``RB``.
"""

IDX_POSITIVE_PARAMS = [
    PARAM_KEYS.index(key) for key in [
        'sigma_int2',
        'sigmac_int2',
        'sigmax2',
        'tau'
    ]
]
"""
Indices into ``PARAM_KEYS`` of parameters that must be strictly positive.

These correspond to variance and scale parameters:
``sigma_int2``, ``sigmac_int2``, ``sigmax2``, and ``tau``.
Used by the emcee prior to enforce positivity constraints.
"""

def to_param_array(hyper_params):
    """
    Convert a dictionary of global hyperparameters to a flat 1-D array.

    Extracts values from ``hyper_params`` in the canonical order defined by
    ``PARAM_KEYS`` and stacks them into a single NumPy array. This is the
    format expected by the emcee sampler (each walker position is a 1-D
    array of length ``len(PARAM_KEYS)``).

    Parameters
    ----------
    hyper_params : dict
        Dictionary mapping parameter names to scalar values. Must contain
        at least the keys listed in ``PARAM_KEYS``.

    Returns
    -------
    np.ndarray, shape (11,)
        Flat array of hyperparameter values in ``PARAM_KEYS`` order.

    See Also
    --------
    from_param_array : Inverse operation.
    """
    return np.array([hyper_params[k] for k in PARAM_KEYS])

def from_param_array(x):
    """
    Convert a flat 1-D hyperparameter array back to a named dictionary.

    Pairs elements of ``x`` with the corresponding names in ``PARAM_KEYS``,
    returning a dictionary suitable for use with the likelihood, prior, and
    intrinsic-distribution utilities.

    Parameters
    ----------
    x : np.ndarray, shape (11,) or (11, N)
        Flat array (or batch of arrays) of hyperparameter values in the
        canonical ``PARAM_KEYS`` order, as produced by :func:`to_param_array`
        or read from an emcee backend.

    Returns
    -------
    dict
        Dictionary mapping each name in ``PARAM_KEYS`` to the corresponding
        element (or row) of ``x``.

    See Also
    --------
    to_param_array : Inverse operation.
    """
    return dict(zip(PARAM_KEYS, x))