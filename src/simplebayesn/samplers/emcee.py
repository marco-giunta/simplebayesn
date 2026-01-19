import numpy as np
import emcee
from multiprocessing import Pool
from ..distributions.likelihood import marginal_loglikelihood as lkl
from ..utils.param_array import (
    PARAM_KEYS,
    IDX_POSITIVE_PARAMS,
    to_param_array,
    from_param_array
)
from ..utils.data import SaltData, SaltDataCompact
from functools import partial
from ..distributions.selection.numpy import log_selection_probability as log_sp_numpy
from ..distributions.selection.jax import (
    log_selection_probability_2,
    log_selection_probability_3,
    log_selection_probability_4,
    log_selection_probability_4b
)
from ..distributions.selection.jax import preprocess_arguments_log_selection_probability as preproc_args_jax
import jax.numpy as jnp

from ..distributions.selection.jax import preproc_args, lsp_mc_vec_jax
def log_posterior_selection_jax_1(x, prior, observed_data: SaltData | SaltDataCompact,
                                  clim: tuple[float], xlim: tuple[float],
                                  num_sim_per_sample: int):
    LP = log_posterior(x, prior, observed_data)
    LSP = lsp_mc_vec_jax(
        **preproc_args(observed_data=observed_data, global_params=from_param_array(x)),
        clim=clim, xlim=xlim,
        num_sim_per_sample=num_sim_per_sample,
        seed=0
    )
    if not np.isfinite(LSP):
        return -np.inf
    return LP - LSP

# def get_log_posterior(prior: callable, observed_data: SaltData) -> callable:
#     return lambda x: lkl(from_param_array(x), observed_data) + prior(from_param_array(x))

def log_posterior(x, prior, observed_data):
    params = from_param_array(x)
    PR = prior(params)
    if not np.isfinite(PR):
        return -np.inf
    LL = lkl(params, observed_data)
    if not np.isfinite(LL): # superfluo se mi assicuro tau>0, sigmax2>0, ecc. usando il prior su R+
        return -np.inf
    return LL + PR

def log_posterior_selection_numpy(x, prior, observed_data: SaltData | SaltDataCompact,
                                  mlim: tuple[float], clim: tuple[float], xlim: tuple[float],
                                  Nm: int, Nc: int, Nx: int):
    LP = log_posterior(x, prior, observed_data)
    LSP = log_sp_numpy(
        global_params=from_param_array(x),
        observed_data=observed_data,
        mlim=mlim, clim=clim, xlim=xlim,
        Nm=Nm, Nc=Nc, Nx=Nx
    )
    if not np.isfinite(LSP):
        return -np.inf
    return LP - LSP

def log_posterior_selection_jax_2(x, prior, observed_data: SaltData | SaltDataCompact,
                                    mlim: tuple[float], clim: tuple[float], xlim: tuple[float],
                                    Nm: int, Nc: int, Nx: int):
        LP = log_posterior(x, prior, observed_data)
        LSP = log_selection_probability_2(
            mlim=mlim, clim=clim, xlim=xlim,
            Nm=Nm, Nc=Nc, Nx=Nx,
            **preproc_args_jax(
                global_params=from_param_array(x),
                observed_data=observed_data
            )
        )
        if not jnp.isfinite(LSP):
            return -np.inf
        return LP - float(LSP)

def log_posterior_selection_jax_3(x, prior, observed_data: SaltData | SaltDataCompact,
                                    mlim: tuple[float], clim: tuple[float], xlim: tuple[float],
                                    Nm: int, Nc: int, Nx: int):
        LP = log_posterior(x, prior, observed_data)
        LSP = log_selection_probability_3(
            mlim=mlim, clim=clim, xlim=xlim,
            Nm=Nm, Nc=Nc, Nx=Nx,
            **preproc_args_jax(
                global_params=from_param_array(x),
                observed_data=observed_data
            )
        )
        if not jnp.isfinite(LSP):
            return -np.inf
        return LP - float(LSP)

def log_posterior_selection_jax_4(x, prior, observed_data: SaltData | SaltDataCompact,
                                    mlim: tuple[float], clim: tuple[float], xlim: tuple[float],
                                    Nm: int, Nc: int, Nx: int):
        LP = log_posterior(x, prior, observed_data)
        LSP = log_selection_probability_4(
            mlim=mlim, clim=clim, xlim=xlim,
            Nm=Nm, Nc=Nc, Nx=Nx,
            **preproc_args_jax(
                global_params=from_param_array(x),
                observed_data=observed_data
            )
        )
        if not jnp.isfinite(LSP):
            return -np.inf
        return LP - float(LSP)

def log_posterior_selection_jax_4b(x, prior, observed_data: SaltData | SaltDataCompact,
                                  mlim: tuple[float], clim: tuple[float], xlim: tuple[float],
                                  Nm: int, Nc: int, Nx: int, batch_size: int):
    LP = log_posterior(x, prior, observed_data)
    LSP = log_selection_probability_4b(
        mlim=mlim, clim=clim, xlim=xlim,
        Nm=Nm, Nc=Nc, Nx=Nx,
        **preproc_args_jax(
            global_params=from_param_array(x),
            observed_data=observed_data
        ),
        batch_size=batch_size
    )
    if not jnp.isfinite(LSP):
        return -np.inf
    return LP - float(LSP)

def emcee_sampler(num_walkers: int, num_burnin: int, num_samples: int,
                  initial_values: np.ndarray,
                  prior: callable, observed_data: SaltData | SaltDataCompact,
                  selection: str = None,
                  mlim: tuple[float] = None, clim: tuple[float] = None, xlim: tuple[float] = None,
                  Nm: int = None, Nc: int = None, Nx: int = None, batch_size: int = None,
                  
                  num_sim_per_sample: int = None,
                  
                  backend = None, resume: bool = False,
                  parallel: bool = False, progress: bool = True):    
    
    lps_dict = {
        'jax1': log_posterior_selection_jax_1,
        'jax2': log_posterior_selection_jax_2,
        'jax3': log_posterior_selection_jax_3,
        'jax4': log_posterior_selection_jax_4,
        'jax4b': log_posterior_selection_jax_4b
    }

    if selection is None:
        log_prob = partial(log_posterior, prior=prior, observed_data=observed_data)
    
    elif selection == 'jax1':
        lps_jax = lps_dict[selection]
        log_prob = partial(lps_jax, prior=prior, observed_data=observed_data,
                           clim=clim, xlim=xlim, num_sim_per_sample=num_sim_per_sample)
    
    elif selection == 'numpy':
        log_prob = partial(log_posterior_selection_numpy, prior=prior, observed_data=observed_data,
                           mlim=mlim, clim=clim, xlim=xlim,
                           Nm=Nm, Nc=Nc, Nx=Nx)
    elif selection in ['jax2', 'jax3', 'jax4']:
        lps_jax = lps_dict[selection]
        log_prob = partial(lps_jax, prior=prior, observed_data=observed_data,
                           mlim=mlim, clim=clim, xlim=xlim,
                           Nm=Nm, Nc=Nc, Nx=Nx)
    elif selection == 'jax4b':
        lps_jax = lps_dict[selection]
        log_prob = partial(lps_jax, prior=prior, observed_data=observed_data,
                           mlim=mlim, clim=clim, xlim=xlim,
                           Nm=Nm, Nc=Nc, Nx=Nx, batch_size=batch_size)
    else:
        raise ValueError(f'invalid {selection=}')

    if parallel:
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(
                num_walkers,
                11,
                log_prob,
                pool = pool,
                backend = backend
            )

            if resume:
                sampler.run_mcmc(None, num_samples, progress = progress)
            else:
                if num_burnin is not None:
                    burnin_state = sampler.run_mcmc(initial_values, num_burnin, progress = progress)
                    sampler.reset()
                    sampler.run_mcmc(burnin_state, num_samples, progress = progress)
                else:
                    sampler.run_mcmc(initial_values, num_samples, progress = progress)

    else:
        sampler = emcee.EnsembleSampler(
            num_walkers,
            11,
            log_prob,
            pool = None,
            backend = backend
        )

        if resume:
            sampler.run_mcmc(None, num_samples, progress = progress)
        else:
            if num_burnin is not None:
                burnin_state = sampler.run_mcmc(initial_values, num_burnin, progress = progress)
                sampler.reset()
                sampler.run_mcmc(burnin_state, num_samples, progress = progress)
            else:
                sampler.run_mcmc(initial_values, num_samples, progress = progress)

    return sampler
