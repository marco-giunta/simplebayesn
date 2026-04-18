import numpy as np
import torch

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

def to_param_array(hyper_params):
    """
    Convert a dictionary of global hyperparameters to a flat 1-D array.

    Extracts values from ``hyper_params`` in the canonical order defined by
    ``PARAM_KEYS`` and stacks them into a single NumPy array.

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

    Pairs elements of ``x`` with the corresponding names in ``PARAM_KEYS``.

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

@torch.no_grad
def from_param_batch(params_batch: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Convert a batch of parameter vectors to a named dictionary of tensors.
 
    The inverse of stacking individual :func:`to_param_array` outputs
    row-wise.  Where :func:`from_param_array` handles a single 1-D array
    and returns a dict of scalars, this function handles the ``(W, 11)``
    array that emcee passes under ``vectorize=True`` and returns a dict of
    ``(W,)`` tensors — one per parameter — suitable for direct use in
    vectorised PyTorch operations.
 
    Parameters
    ----------
    params_batch : torch.Tensor, shape (W, 11)
        W parameter vectors stacked row-wise, in canonical
        :data:`PARAM_KEYS` order.
 
    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary mapping each parameter name in :data:`PARAM_KEYS` to
        a ``(W,)`` tensor of its values across the W walkers.
 
    See Also
    --------
    from_param_array : Single-vector equivalent returning scalar values.
    to_param_array : Inverse operation for single vectors.
 
    Examples
    --------
    ::
 
        X = torch.randn(32, 11)          # 32 walker proposals
        params = from_param_batch(X)
        tau = params['tau']              # shape (32,)
        RB  = params['RB']               # shape (32,)
    """
    return {param: params_batch[:, i] for i, param in enumerate(PARAM_KEYS)}