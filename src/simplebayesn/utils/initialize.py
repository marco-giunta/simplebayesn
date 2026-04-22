import numpy as np
from ..utils.param_array import to_param_array as to_param_array_fun

def get_default_ranges(allow_negative_beta_int: bool = False) -> dict:
    """
    Return the default prior sampling ranges for all model parameters.

    Provides conservative uniform intervals for drawing initial values for
    both global hyperparameters and per-SN latent parameters. These ranges
    are intended to produce starting points that are broadly consistent with
    typical Type Ia supernova populations, without being overly informative.

    Parameters
    ----------
    allow_negative_beta_int: bool, default False
        if True, the default beta_int range will be (-1, 1), otherwise (2.1, 2.3).

    Returns
    -------
    dict
        A nested dictionary with two top-level keys:

        ``'latent_params'``
            dict mapping each latent parameter name to a ``(low, high)``
            tuple for uniform sampling:

            - ``'E'``        : dust reddening ``(0, 0.2)``
            - ``'m_app'``    : apparent magnitude ``(15, 20)``
            - ``'c_app'``    : apparent color ``(-0.3, 0.3)``
            - ``'x'``        : stretch ``(-3, 3)``
            - ``'dist_mod'`` : distance modulus ``(30, 40)``

        ``'global_params'``
            dict mapping each hyperparameter name to a ``(low, high)``
            tuple for uniform sampling:

            - ``'tau'``        : dust scale ``(0.03, 0.2)``
            - ``'RB'``         : dust ratio ``(3, 5)``
            - ``'x0'``         : mean stretch ``(-0.5, -0.3)``
            - ``'sigmax2'``    : stretch variance ``(1.0, 2.0)``
            - ``'c0_int'``     : mean intrinsic color ``(-0.1, 0.0)``
            - ``'alphac_int'`` : color-stretch slope ``(-1.0, 0.0)``
            - ``'sigmac_int2'``: intrinsic color variance ``(0.001, 0.01)``
            - ``'M0_int'``     : mean intrinsic magnitude ``(-20, -18)``
            - ``'alpha'``      : intrinsic stretch-magnitude ``(-0.16, -0.14)``
            - ``'beta_int'``   : intrinsic color-magnitude slope ``(2.1, 2.3) or (-1, 1)``
            - ``'sigma_int2'`` : intrinsic scatter variance ``(0.01, 0.2)``
    """
    return {
        'latent_params': {
            'E': (0, 0.2),
            'm_app': (15, 20),
            'c_app': (-0.3, 0.3),
            'x': (-3, 3),
            'dist_mod': (30, 40)
        },

        'global_params': {
            'tau': (0.03, 0.2),
            'RB': (3, 5),
            'x0': (-0.5, -0.3),
            'sigmax2': (1., 2.),
            'c0_int': (-0.1, 0.),
            'alphac_int': (-1., 0.),
            'sigmac_int2': (0.001, 0.01),
            'M0_int': (-20, -18),
            'alpha': (-0.16, -0.14),
            'beta_int': (-1, 1) if allow_negative_beta_int else (2.1, 2.3),
            'sigma_int2': (0.01, 0.2)
        }
    }

def _sample_initial_values_uniform(num_samples: int, seed: int = None,
                                   ranges_dict: dict = None,
                                   marginal: bool = False,
                                   allow_negative_beta_int: bool = False) -> dict:
    """
    Draw a single set of initial values from uniform distributions.

    Internal helper called by :func:`sample_initial_values_uniform`. Samples
    scalar initial values for each global hyperparameter and length-``num_samples``
    arrays for each latent parameter, independently and uniformly within the
    ranges specified by ``ranges_dict``.

    Parameters
    ----------
    num_samples : int
        Number of supernovae; sets the length of each latent-parameter array.
    seed : int or None, optional
        Seed for the NumPy random number generator. If ``None``, a random
        seed is used.
    ranges_dict : dict or None, optional
        Nested dictionary of ``(low, high)`` ranges in the same format as
        returned by :func:`get_default_ranges`. If ``None``, the default
        ranges are used.
    marginal : bool, optional
        If ``True``, return only the ``'global_params'`` sub-dictionary,
        omitting the latent parameters. Default is ``False``.
    allow_negative_beta_int: bool, default False
        if True, the default beta_int range will be (-1, 1), otherwise (2.1, 2.3).

    Returns
    -------
    dict
        If ``marginal=False``: a nested dict with keys ``'latent_params'``
        and ``'global_params'``, each mapping parameter names to their
        sampled initial values.
        If ``marginal=True``: only the ``'global_params'`` sub-dictionary.
    """
    rng = np.random.default_rng(seed)
    if ranges_dict is None:
        ranges_dict = get_default_ranges(
            allow_negative_beta_int = allow_negative_beta_int
        )
    
    lp = ranges_dict['latent_params']
    gp = ranges_dict['global_params']
    n = num_samples

    iv = {
        'latent_params': {
            'E':rng.uniform(*lp['E'], n),
            'm_app':rng.uniform(*lp['m_app'], n),
            'c_app':rng.uniform(*lp['c_app'], n),
            'x':rng.uniform(*lp['x'], n),
            'dist_mod':rng.uniform(*lp['dist_mod'], n),
        },
        'global_params': {
            'tau':rng.uniform(*gp['tau']),
            'RB':rng.uniform(*gp['RB']),
            'x0':rng.uniform(*gp['x0']),
            'sigmax2':rng.uniform(*gp['sigmax2']),
            'c0_int':rng.uniform(*gp['c0_int']),
            'alphac_int':rng.uniform(*gp['alphac_int']),
            'sigmac_int2':rng.uniform(*gp['sigmac_int2']),
            'M0_int':rng.uniform(*gp['M0_int']),
            'alpha':rng.uniform(*gp['alpha']),
            'beta_int':rng.uniform(*gp['beta_int']),
            'sigma_int2':rng.uniform(*gp['sigma_int2']),
        }
    }
    return iv if not marginal else iv['global_params']

def sample_initial_values_uniform(num_samples: int,
                                  seed: int | list[int] = None,
                                  ranges_dict: dict = None,
                                  marginal: bool = False,
                                  to_param_array: bool = False,
                                  allow_negative_beta_int: bool = False):
    """
    Sample initial parameter values uniformly within prior-consistent ranges.

    Generates starting points for the MCMC samplers (Gibbs or emcee) by
    drawing independently from uniform distributions within the ranges given
    by ``ranges_dict``. Supports both single draws and batched draws for
    multiple walkers (e.g. for emcee).

    Parameters
    ----------
    num_samples : int
        Number of supernovae; determines the size of each latent-parameter
        array.
    seed : int, iterable of int, or None, optional
        Random seed(s). If an iterable is provided, one independent draw is
        produced per seed value, enabling multi-walker initialisation for
        emcee. If ``None``, a random seed is used for each draw.
    ranges_dict : dict or None, optional
        Custom sampling ranges in the format of :func:`get_default_ranges`.
        If ``None``, the default ranges are used.
    marginal : bool, optional
        If ``True``, return only global hyperparameters (no latent arrays).
        Required when ``to_param_array=True``. Default is ``False``.
    to_param_array : bool, optional
        If ``True``, convert the global-params dictionary (or list thereof)
        to a flat NumPy array (or 2-D array for multiple seeds) using
        :func:`~simplebayesn.utils.param_array.to_param_array`. Can only be
        used when ``marginal=True``. Default is ``False``.
    allow_negative_beta_int: bool, default False
        if True, the default beta_int range will be (-1, 1), otherwise (2.1, 2.3).

    Returns
    -------
    dict or list of dict or np.ndarray
        - If ``seed`` is a scalar (or ``None``): a single dict (nested if
          ``marginal=False``, flat global-params dict if ``marginal=True``)
          or a 1-D array of length 11 if ``to_param_array=True``.
        - If ``seed`` is an iterable: a list of dicts, or a 2-D array of
          shape ``(len(seed), 11)`` if ``to_param_array=True``.

    Raises
    ------
    ValueError
        If ``to_param_array=True`` but ``marginal=False``, since latent arrays
        cannot be meaningfully flattened into the global-parameter vector.

    Examples
    --------
    Single Gibbs initialisation::

        iv = sample_initial_values_uniform(num_samples=100, seed=42)
        # iv['global_params']['tau'], iv['latent_params']['E'], ...

    Batch initialisation for emcee (32 walkers)::

        p0 = sample_initial_values_uniform(
            num_samples=100, seed=range(32),
            marginal=True, to_param_array=True
        )  # shape (32, 11)
    """
    if hasattr(seed, '__iter__'):
        iv = [_sample_initial_values_uniform(
            num_samples=num_samples,
            seed=n,
            ranges_dict=ranges_dict,
            marginal=marginal,
            allow_negative_beta_int=allow_negative_beta_int
        ) for n in seed]
        if to_param_array:
            if marginal is False:
                raise ValueError(f'to_param_array is True but marginal is False')
            iv = np.array([to_param_array_fun(i) for i in iv])
        return iv
    else:
        iv = _sample_initial_values_uniform(
            num_samples=num_samples,
            seed=seed,
            ranges_dict=ranges_dict,
            marginal=marginal,
            allow_negative_beta_int=allow_negative_beta_int
        )
        if to_param_array:
            if marginal is False:
                raise ValueError(f'to_param_array is True but marginal is False')
            iv = to_param_array_fun(iv)
        return iv