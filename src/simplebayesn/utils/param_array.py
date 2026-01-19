import numpy as np

PARAM_KEYS = [
    'M0_int', 'alpha', 'beta_int', 'sigma_int2',
    'c0_int', 'alphac_int', 'sigmac_int2', 'x0',
    'sigmax2', 'tau', 'RB'
]

IDX_POSITIVE_PARAMS = [
    PARAM_KEYS.index(key) for key in [
        'sigma_int2',
        'sigmac_int2',
        'sigmax2',
        'tau'
    ]
]

def to_param_array(hyper_params):
    return np.array([hyper_params[k] for k in PARAM_KEYS])

def from_param_array(x):
    return dict(zip(PARAM_KEYS, x))