from .data import GibbsChainDataCompact
from .param_array import from_param_array
from emcee.backends import HDFBackend
from pathlib import Path

def from_emcee_h5_to_gibbs_chain_data(h5path: str | Path):
    reader = HDFBackend(h5path)
    global_params = from_param_array(reader.get_chain(flat=True).T)
    N = len(global_params['tau'])
    latent_params = {
        'm_app':np.zeros(N),
        'c_app':np.zeros(N),
        'x':np.zeros(N),
        'E':np.zeros(N),
        'dist_mod':np.zeros(N)
    }
    return GibbsChainDataCompact(**global_params, **latent_params)
