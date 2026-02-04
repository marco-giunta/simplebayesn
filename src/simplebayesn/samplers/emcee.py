import numpy as np
import emcee
from multiprocessing import Pool
from ..distributions.likelihood import marginal_loglikelihood as log_lkl
from ..utils.param_array import from_param_array
from ..utils.data import SaltData, SaltDataCompact
from functools import partial
from ..distributions.selection.mc import (
    preprocess_arguments_log_selection_probability_mc_jax,
    log_selection_probability_mc_jax
)

def log_posterior(x, log_prior, observed_data):
    params = from_param_array(x)
    LPR = log_prior(params)
    if not np.isfinite(LPR):
        return -np.inf
    LL = log_lkl(params, observed_data)
    if not np.isfinite(LL): # superfluo se mi assicuro tau>0, sigmax2>0, ecc. usando il log_prior su R+
        return -np.inf
    return LL + LPR

def log_posterior_selection(x, log_prior, observed_data: SaltData | SaltDataCompact,
                            clim: tuple[float], xlim: tuple[float],
                            num_sim_per_sample: int):
    LP = log_posterior(x, log_prior, observed_data)
    LSP = log_selection_probability_mc_jax(
        **preprocess_arguments_log_selection_probability_mc_jax(observed_data=observed_data,
                                                                global_params=from_param_array(x)),
        clim=clim, xlim=xlim,
        num_sim_per_sample=num_sim_per_sample,
        seed=0
    )
    if not np.isfinite(LSP):
        return -np.inf
    return LP - LSP

def emcee_sampler(num_walkers: int, num_burnin: int, num_samples: int,
                  initial_values: np.ndarray,
                  log_prior: callable, observed_data: SaltData | SaltDataCompact,
                  selection: bool = False,
                  clim: tuple[float] = None, xlim: tuple[float] = None,                  
                  num_sim_per_sample: int = None,                  
                  backend = None, resume: bool = False,
                  parallel: bool = False, progress: bool = True):    

    if not selection:
        log_prob = partial(log_posterior, log_prior=log_prior, observed_data=observed_data)
    
    else:
        log_prob = partial(log_posterior_selection, log_prior=log_prior, observed_data=observed_data,
                           clim=clim, xlim=xlim, num_sim_per_sample=num_sim_per_sample)

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