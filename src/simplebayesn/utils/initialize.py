import numpy as np
from ..utils.param_array import to_param_array as to_param_array_fun

def get_default_ranges():
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
            'beta_int': (2.1, 2.3),
            'sigma_int2': (0.01, 0.2)
        }
    }

def _sample_initial_values_uniform(num_samples: int, seed: int = None,
                                   ranges_dict: dict = None,
                                   marginal: bool = False):
    rng = np.random.default_rng(seed)
    if ranges_dict is None:
        ranges_dict = get_default_ranges()
    
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
                                  to_param_array: bool = False):
    if hasattr(seed, '__iter__'):
        iv = [_sample_initial_values_uniform(
            num_samples=num_samples,
            seed=n,
            ranges_dict=ranges_dict,
            marginal=marginal
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
            marginal=marginal
        )
        if to_param_array:
            if marginal is False:
                raise ValueError(f'to_param_array is True but marginal is False')
            iv = to_param_array_fun(iv)
        return iv