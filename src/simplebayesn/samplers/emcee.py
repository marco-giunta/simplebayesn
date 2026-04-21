import numpy as np
import emcee
from pathlib import Path
from ..distributions.likelihood import marginal_loglikelihood_vectorized
from ..utils.param_array import from_param_array
from ..utils.data import SaltData
from functools import partial
import torch
from ..distributions.selection.mc import log_selection_prob_mc_vectorized
from ..distributions.selection.grid import log_selection_prob_grid_vectorized

def log_posterior_vectorized(
        X: np.ndarray,
        log_prior: callable,
        observed_data: SaltData,
        dtype: torch.dtype = torch.float32,
) -> np.ndarray:
    """
    Vectorized log-posterior for a batch of walker proposals (no selection correction).
 
    Receives the ``(W, 11)`` numpy array that emcee passes under
    ``vectorize=True`` and returns a ``(W,)`` numpy array of log-posterior
    values.
 
    The log-prior is evaluated on CPU for each walker individually (a small
    Python loop over cheap arithmetic - negligible cost).  Walkers that fail
    the prior check are masked out before the GPU likelihood call so no
    computation is wasted on them.  The marginal log-likelihood is always
    evaluated on CPU regardless of ``dtype``, since batched small-matrix
    operations are faster there.
 
    Parameters
    ----------
    X : np.ndarray, shape (W, 11)
        Walker proposals in canonical
        :data:`~simplebayesn.utils.param_array.PARAM_KEYS` order, as passed
        by emcee under ``vectorize=True``.
    log_prior : callable
        Log-prior function ``log_prior(params: dict) -> float``.
    observed_data : SaltData
        Preprocessed supernova dataset.
    dtype : torch.dtype, optional
        Floating-point precision for the likelihood computation.
        Default is ``torch.float32``.
 
    Returns
    -------
    np.ndarray, shape (W,)
        Log-posterior values.  Entries are ``-inf`` for walkers rejected
        by the prior or producing a non-finite likelihood.
    """
    W = X.shape[0]
    log_priors = np.array(  # (W,)
        [log_prior(from_param_array(X[w])) for w in range(W)]
    ) # small loop on CPU + simple functions = cheap enough

    log_psts = np.full(W, -np.inf)
    idx_prior = np.where(np.isfinite(log_priors))[0]  # integer indices into W
    if idx_prior.size == 0:
        return log_psts

    log_lkls = marginal_loglikelihood_vectorized(  # (len(idx_prior),)
        X[idx_prior], observed_data, device='cpu', dtype=dtype
    ).numpy() # base likelihood doesn't benefit from cuda

    idx_ll = idx_prior[np.isfinite(log_lkls)]  # back into original W indexing
    log_psts[idx_ll] = log_priors[idx_ll] + log_lkls[np.isfinite(log_lkls)]

    return log_psts

def log_posterior_selection_mc_vectorized(
        X: np.ndarray,
        log_prior: callable,
        observed_data: SaltData,
        clim: tuple[float, float],
        xlim: tuple[float, float],
        num_sim_per_sample: int,
        seed: int = 0,
        device: torch.device | str = 'cuda',
        dtype: torch.dtype = torch.float32,
) -> np.ndarray:
    """
    Vectorized selection-corrected log-posterior using MC estimation.
 
    Extends :func:`log_posterior_vectorized` by subtracting the MC-estimated
    total log selection probability:
 
    .. math::
 
        \\log \\tilde{p}(\\theta_w \\mid \\text{data})
        = \\log p(\\theta_w) + \\log p(\\text{data} \\mid \\theta_w)
          - \\sum_n \\log \\hat{p}_n^{\\rm MC}(\\theta_w)
 
    Three successive filters are applied to avoid unnecessary computation:
    walkers that fail the prior are excluded before the likelihood call,
    and walkers with a non-finite likelihood are excluded before the (more
    expensive) selection probability call.
 
    The marginal likelihood is evaluated on CPU; the MC selection correction
    is evaluated on ``device`` (GPU by default), where the large
    ``(W, N, K)`` forward simulation is efficiently vectorized.
 
    Parameters
    ----------
    X : np.ndarray, shape (W, 11)
        Walker proposals in canonical
        :data:`~simplebayesn.utils.param_array.PARAM_KEYS` order.
    log_prior : callable
        Log-prior function.
    observed_data : SaltData
        Preprocessed supernova dataset.
    clim : tuple of float
        ``(min, max)`` colour cut on the simulated observed colour.
    xlim : tuple of float
        ``(min, max)`` stretch cut on the simulated observed stretch.
    num_sim_per_sample : int
        Number of forward-model draws per SN per walker (K).
    seed : int, optional
        Fixed random seed for the MC generator.  Must remain constant
        across emcee calls to ensure the log-posterior is deterministic.
        Default is 0.
    device : torch.device or str, optional
        Device for the MC selection computation.  Default is ``'cuda'``.
    dtype : torch.dtype, optional
        Floating-point precision.  Default is ``torch.float32``.
 
    Returns
    -------
    np.ndarray, shape (W,)
        Selection-corrected log-posterior values.  Entries are ``-inf`` for
        walkers rejected at any stage (prior, likelihood, or selection).
    """
    W = X.shape[0]
    log_priors = np.array(
        [log_prior(from_param_array(X[w])) for w in range(W)]
    )  # (W,)

    log_psts = np.full(W, -np.inf)
    idx_prior = np.where(np.isfinite(log_priors))[0]
    if idx_prior.size == 0:
        return log_psts

    log_lkls = marginal_loglikelihood_vectorized(
        X[idx_prior], observed_data, device='cpu', dtype=dtype
    ).numpy()  # (len(idx_prior),)

    finite_ll = np.isfinite(log_lkls)
    idx_ll = idx_prior[finite_ll]  # indices of walkers that passed both checks
    if idx_ll.size == 0:
        return log_psts

    log_sel_probs = log_selection_prob_mc_vectorized(
        X[idx_ll], observed_data,
        clim=clim, xlim=xlim,
        num_sim_per_sample=num_sim_per_sample,
        seed=seed, device=device, dtype=dtype,
    ).cpu().numpy()  # (len(idx_ll),)

    finite_lsp = np.isfinite(log_sel_probs)
    idx_lsp = idx_ll[finite_lsp]  # indices that passed all three checks

    log_psts[idx_lsp] = (
        log_priors[idx_lsp]
        + log_lkls[finite_ll][finite_lsp]
        + (- log_sel_probs[finite_lsp])
    )

    return log_psts

def log_posterior_selection_grid_vectorized(
        X: np.ndarray,
        log_prior: callable,
        observed_data: SaltData,
        mlim: tuple[float, float],
        clim: tuple[float, float],
        xlim: tuple[float, float],
        Nm: int,
        Nc: int,
        Nx: int,
        sn_batch_size: int = 32,
        device: torch.device | str = 'cuda',
        dtype: torch.dtype | str = torch.float32
):
    """
    Vectorised selection-corrected log-posterior using grid integration.
 
    Extends :func:`log_posterior_vectorized` by subtracting the grid-integrated
    total log selection probability:
 
    .. math::
 
        \\log \\tilde{p}(\\theta_w \\mid \\text{data})
        = \\log p(\\theta_w) + \\log p(\\text{data} \\mid \\theta_w)
          - \\sum_n \\log S_n^{\\rm grid}(\\theta_w)
 
    Unlike the MC variant, the grid correction is **deterministic** - the
    same parameter vector always produces the same correction value, with no
    Monte Carlo noise.  This is preferable when the observable box is
    well-defined and low noise in the log-posterior is important.
 
    The same three-stage filtering as in
    :func:`log_posterior_selection_mc_vectorized` is applied to avoid
    unnecessary computation.
 
    .. note::
        There is a subtle difference from the MC version in the current
        implementation: ``log_sel_probs`` is computed from the full ``X``
        array rather than ``X[idx_ll]``.  This means the grid selection
        integral is evaluated for all W walkers, including those already
        known to have ``-inf`` posterior.  This is harmless (those entries
        are overwritten anyway) but slightly wasteful.  A future version
        should pass ``X[idx_ll]`` and index back consistently, as in the
        MC version.
 
    Parameters
    ----------
    X : np.ndarray, shape (W, 11)
        Walker proposals in canonical
        :data:`~simplebayesn.utils.param_array.PARAM_KEYS` order.
    log_prior : callable
        Log-prior function.
    observed_data : SaltData
        Preprocessed supernova dataset.
    mlim : tuple of float
        ``(min, max)`` apparent magnitude limits.
    clim : tuple of float
        ``(min, max)`` colour limits.
    xlim : tuple of float
        ``(min, max)`` stretch limits.
    Nm, Nc, Nx : int
        Grid resolution along each axis.
    sn_batch_size : int, optional
        Number of SNe per loop iteration in the grid computation.
        Default is 32.
    device : torch.device or str, optional
        Device for the grid computation.  Default is ``'cuda'``.
    dtype : torch.dtype, optional
        Floating-point precision.  Default is ``torch.float32``.
 
    Returns
    -------
    np.ndarray, shape (W,)
        Selection-corrected log-posterior values.  Entries are ``-inf`` for
        walkers rejected at any stage.
    """
    W = X.shape[0]
    log_priors = np.array(
        [log_prior(from_param_array(X[w])) for w in range(W)]
    )  # (W,)

    log_psts = np.full(W, -np.inf)
    idx_prior = np.where(np.isfinite(log_priors))[0]
    if idx_prior.size == 0:
        return log_psts

    log_lkls = marginal_loglikelihood_vectorized(
        X[idx_prior], observed_data, device='cpu', dtype=dtype
    ).numpy()  # (len(idx_prior),)

    finite_ll = np.isfinite(log_lkls)
    idx_ll = idx_prior[finite_ll]  # indices of walkers that passed both checks
    if idx_ll.size == 0:
        return log_psts

    log_sel_probs = log_selection_prob_grid_vectorized(
        X[idx_ll], observed_data,
        mlim = mlim, clim = clim, xlim = xlim,
        Nm = Nm, Nc = Nc, Nx = Nx,
        sn_batch_size = sn_batch_size,
        device = device, dtype = dtype
    ).cpu().numpy()

    finite_lsp = np.isfinite(log_sel_probs)
    idx_lsp = idx_ll[finite_lsp]  # indices that passed all three checks

    log_psts[idx_lsp] = (
        log_priors[idx_lsp]
        + log_lkls[finite_ll][finite_lsp]
        + (- log_sel_probs[finite_lsp])
    )

    return log_psts

def emcee_sampler(
        num_walkers: int,
        num_burnin: int | None,
        num_samples: int,
        initial_values: np.ndarray,
        log_prior: callable,
        observed_data: SaltData,
        selection: str | None = None,
        clim: tuple[float, float] = None,
        xlim: tuple[float, float] = None,
        # MC-specific
        num_sim_per_sample: int = None,
        mc_seed: int = 0,
        # Grid-specific
        mlim: tuple[float, float] = None,
        Nm: int = 50,
        Nc: int = 50,
        Nx: int = 50,
        sn_batch_size: int = 32,
        # emcee infrastructure
        path: Path | str = None,
        resume: bool = False,
        progress: bool = True,
        # torch infrastructure
        device: torch.device | str = 'cuda',
        dtype: torch.dtype | str = torch.float32
) -> emcee.EnsembleSampler:
    """
    Run the emcee ensemble sampler with GPU-vectorised likelihood evaluation.
 
    Uses emcee's ``vectorize=True`` mode so that all ``num_walkers`` proposals
    are passed to ``log_prob`` as a single ``(W, 11)`` numpy array at each
    step, evaluated in one batched GPU call, and returned as a ``(W,)`` array.
    This eliminates the W sequential Python->GPU round-trips that would arise
    from serial or multiprocessing evaluation.
 
    No ``pool`` is used.  All parallelism over walkers is achieved by
    vectorising the computation across the W walker dimension simultaneously.
 
    Selection correction
    --------------------
    The ``selection`` argument controls whether and how the log-posterior is
    corrected for survey selection bias:
 
    ``selection=None`` (default)
        No correction.  The log-posterior is
        :func:`log_posterior_vectorized`.
 
    ``selection='mc'``
        Monte Carlo correction via
        :func:`log_posterior_selection_mc_vectorized`.  Stochastic but
        memory-efficient.  Fix ``mc_seed`` to keep the log-posterior
        deterministic across calls (required by emcee).
        Requires ``clim``, ``xlim``, ``num_sim_per_sample``.
 
    ``selection='grid'``
        Deterministic grid integration via
        :func:`log_posterior_selection_grid_vectorized`.  No MC noise.
        Requires ``clim``, ``xlim``, ``mlim``.
 
    Computation devices
    -------------------
    The marginal likelihood is always computed on CPU (``device='cpu'``
    is passed internally to :func:`marginal_loglikelihood_vectorized`)
    regardless of the ``device`` argument.  The selection correction is
    computed on ``device`` (GPU by default).  This split reflects the
    different performance characteristics of each computation: the
    ``(W, N, 3, 3)`` small-matrix operations in the likelihood are faster
    on CPU, while the large elementwise operations in the selection
    correction benefit from GPU parallelism.
 
    Burn-in handling
    ----------------
    If ``num_burnin`` is provided and ``resume=False``, the sampler runs
    ``num_burnin`` steps, resets the chain, then runs ``num_samples``
    production steps.  If ``num_burnin`` is ``None``, the full
    ``num_samples`` steps are run without a separate burn-in phase.
 
    Parameters
    ----------
    num_walkers : int
        Number of emcee ensemble walkers.
    num_burnin : int or None
        Burn-in steps to discard before the production run.  ``None``
        skips the explicit burn-in phase.
    num_samples : int
        Number of production MCMC steps.
    initial_values : np.ndarray, shape (num_walkers, 11)
        Initial walker positions in canonical
        :data:`~simplebayesn.utils.param_array.PARAM_KEYS` order.
        Typically produced by
        :func:`~simplebayesn.utils.initialize.sample_initial_values_uniform`
        with ``marginal=True``, ``to_param_array=True``.
    log_prior : callable
        Log-prior function ``log_prior(params: dict) -> float``.
    observed_data : SaltData
        Preprocessed supernova dataset.
    selection : str or None, optional
        Selection correction method: ``'mc'``, ``'grid'``, or ``None``.
        Default is ``None``.
    clim : tuple of float or None
        ``(min, max)`` colour limits.  Required when ``selection`` is not
        ``None``.
    xlim : tuple of float or None
        ``(min, max)`` stretch limits.  Required when ``selection`` is not
        ``None``.
    num_sim_per_sample : int or None
        MC draws per SN per walker.  Required when ``selection='mc'``.
    mc_seed : int, optional
        Fixed seed for the MC generator.  Must remain constant across
        calls for deterministic behaviour.  Default is 0.
    mlim : tuple of float or None
        ``(min, max)`` apparent magnitude limits.  Required when
        ``selection='grid'``.
    Nm, Nc, Nx : int, optional
        Grid resolution along each axis.  Default is 50.
    sn_batch_size : int, optional
        SNe per loop iteration in the grid computation.  Default is 32.
    path : str or Path or None, optional
        path to emcee HDF5 backend for checkpointing.  If ``None``,
        the chain is stored only in memory.
    resume : bool, optional
        If ``True``, continue from the state stored in ``backend``;
        ``initial_values`` and ``num_burnin`` are ignored.  Default is
        ``False``.
    progress : bool, optional
        Whether to display a tqdm progress bar.  Default is ``True``.
    device : torch.device or str, optional
        Device for the selection correction computation.  Default is
        ``'cuda'``.
    dtype : torch.dtype, optional
        Floating-point precision for GPU computations.  Default is
        ``torch.float32``, which is significantly faster on consumer GPUs.
 
    Returns
    -------
    emcee.EnsembleSampler
        The sampler object after the run.  Access the flattened chain via
        ``sampler.get_chain(flat=True, discard=num_burnin)`` or via an
        HDF5 backend.
 
    Raises
    ------
    ValueError
        If ``selection`` is not ``None``, ``'mc'``, or ``'grid'``; or if
        required arguments for the chosen selection method are missing.
 
    Examples
    --------
    No selection correction::
 
        sampler = emcee_sampler(
            num_walkers=32, num_burnin=500, num_samples=5000,
            initial_values=p0,
            log_prior=uniform_marginal_log_prior,
            observed_data=data,
        )
 
    MC selection correction::
 
        sampler = emcee_sampler(
            num_walkers=32, num_burnin=500, num_samples=5000,
            initial_values=p0,
            log_prior=uniform_marginal_log_prior,
            observed_data=data,
            selection='mc',
            clim=(-0.2, 0.8), xlim=(-3, 3),
            num_sim_per_sample=1000,
        )
 
    Grid selection correction::
 
        sampler = emcee_sampler(
            num_walkers=32, num_burnin=500, num_samples=5000,
            initial_values=p0,
            log_prior=uniform_marginal_log_prior,
            observed_data=data,
            selection='grid',
            mlim=(14, 22), clim=(-0.2, 0.8), xlim=(-3, 3),
        )
    """
    if selection is not None:
        if selection not in ('mc', 'grid'):
            raise ValueError(
                f"selection must be 'mc' or 'grid', got '{selection}'"
            )
        if clim is None or xlim is None:
            raise ValueError("clim and xlim are required when selection=True")
        if selection == 'mc' and num_sim_per_sample is None:
            raise ValueError(
                "num_sim_per_sample is required when selection='mc'"
            )
        if selection == 'grid' and mlim is None:
            raise ValueError("mlim is required when selection='grid'")


    # vectorized log_prob: (W, 11) -> (W,)
    if not selection:
        log_prob = partial(
            log_posterior_vectorized,
            log_prior=log_prior,
            observed_data=observed_data,
            dtype=dtype
        )
    elif selection == 'mc':
        log_prob = partial(
            log_posterior_selection_mc_vectorized,
            log_prior=log_prior,
            observed_data=observed_data,
            clim=clim, xlim=xlim,
            num_sim_per_sample=num_sim_per_sample,
            seed=mc_seed,
            device=device,
            dtype=dtype
        )
    else:  # 'grid'
        log_prob = partial(
            log_posterior_selection_grid_vectorized,
            log_prior=log_prior,
            observed_data=observed_data,
            mlim=mlim, clim=clim, xlim=xlim,
            Nm=Nm, Nc=Nc, Nx=Nx,
            sn_batch_size=sn_batch_size,
            device=device,
            dtype=dtype
        )

    if path is not None:
        backend = emcee.backends.HDFBackend(Path(path))
    else:
        backend = None

    sampler = emcee.EnsembleSampler(
        num_walkers, 11, log_prob,
        vectorize=True,       # emcee passes (W, 11); expects (W,) back
        #pool=None,            # no multiprocessing, GPU handles parallelism
        backend=backend,
    )

    if resume:
        sampler.run_mcmc(None, num_samples, progress=progress)
    else:
        if num_burnin is not None:
            state = sampler.run_mcmc(initial_values, num_burnin, progress=progress)
            sampler.reset()
            sampler.run_mcmc(state, num_samples, progress=progress)
        else:
            sampler.run_mcmc(initial_values, num_samples, progress=progress)

    return sampler