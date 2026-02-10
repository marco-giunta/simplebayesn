import numpy as np
import emcee
from multiprocessing import Pool
from ..distributions.likelihood import marginal_loglikelihood as log_lkl
from ..utils.param_array import from_param_array
from ..utils.data import SaltData
from functools import partial
from ..distributions.selection.mc import (
    preprocess_arguments_log_selection_probability_mc_jax,
    log_selection_probability_mc_jax,
    get_kde_interpolant_grids
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

def log_posterior_selection(x, log_prior, observed_data: SaltData,
                            clim: tuple[float], xlim: tuple[float],
                            num_sim_per_sample: int,
                            use_kde_selection: bool,
                            m_grid, c_grid, z_grid, sel_prob_grid):
    LP = log_posterior(x, log_prior, observed_data)
    LSP = log_selection_probability_mc_jax(
        **preprocess_arguments_log_selection_probability_mc_jax(observed_data=observed_data,
                                                                global_params=from_param_array(x)),
        clim=clim, xlim=xlim,
        num_sim_per_sample=num_sim_per_sample,
        use_kde_selection=use_kde_selection,
        m_grid=m_grid, c_grid=c_grid, z_grid=z_grid, sel_prob_grid=sel_prob_grid,
        seed=0
    )
    if not np.isfinite(LSP):
        return -np.inf
    return LP - LSP

def emcee_sampler(num_walkers: int, num_burnin: int, num_samples: int,
                  initial_values: np.ndarray,
                  log_prior: callable, observed_data: SaltData,
                  selection: bool | None = None, kde_args: dict | None = None,
                  clim: tuple[float] = None, xlim: tuple[float] = None,                  
                  num_sim_per_sample: int = None,                  
                  backend = None, resume: bool = False,
                  parallel: bool = False, progress: bool = True):    

    if selection is None:
        log_prob = partial(log_posterior, log_prior=log_prior, observed_data=observed_data)
    
    else:
        if selection == 'cuts':
            use_kde_selection = False
            m_grid = c_grid = z_grid = sel_prob_grid = None
        elif selection == 'kde':
            use_kde_selection = True
            
            kde_args.setdefault('nm', 100)
            kde_args.setdefault('nc', 100)
            kde_args.setdefault('nz', 100)
            kde_args.setdefault('eps', 1e-8)
            kde_args.setdefault('n_mc_norm', 100000)
            
            m_grid, c_grid, z_grid, sel_prob_grid = get_kde_interpolant_grids(
                observed_data.m_app, observed_data.c_app, observed_data.z,
                kde_args['m_app'], kde_args['c_app'], kde_args['z'],
                kde_args['nm'], kde_args['nc'], kde_args['nz'],
                kde_args['eps'], kde_args['n_mc_norm']
            )
        else:
            raise ValueError(f'selection must be "kde", "cuts", or None, got {selection} instead')
            
        log_prob = partial(
            log_posterior_selection,
            log_prior=log_prior,
            observed_data=observed_data,
            clim=clim, xlim=xlim,
            num_sim_per_sample=num_sim_per_sample,
            use_kde_selection=use_kde_selection,
            m_grid=m_grid,
            c_grid=c_grid,
            z_grid=z_grid,
            sel_prob_grid=sel_prob_grid
        )
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