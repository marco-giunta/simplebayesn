import numpy as np
from dataclasses import dataclass, field
import h5py
from pathlib import Path
from emcee.backends import HDFBackend
from .param_array import from_param_array

@dataclass(frozen=True)
class SaltData:
    """
    Immutable container for preprocessed SALT2 supernova light-curve data.

    Stores observed photometric and spectroscopic quantities alongside their
    measurement covariance matrices for a sample of Type Ia supernovae. The
    inverse covariance matrices are computed automatically on construction.

    Parameters
    ----------
    m_app : np.ndarray, shape (N,)
        Apparent magnitude derived from the SALT2 flux parameter x0 via
        ``m_app = -2.5 * log10(x0) + offset``.
    c_app : np.ndarray, shape (N,)
        Observed (apparent) color parameter from the SALT2 fit.
    x : np.ndarray, shape (N,)
        Observed stretch parameter (x1) from the SALT2 fit.
    z : np.ndarray, shape (N,)
        Observed spectroscopic redshift.
    sigma_z : np.ndarray, shape (N,)
        Reported error on the redshift measurement.
    dist_mod : np.ndarray, shape (N,)
        Distance modulus computed from the assumed cosmology at redshift ``z``.
    sigma_mu_z2 : np.ndarray, shape (N,)
        Variance on the distance modulus contribution from redshift uncertainty
        and peculiar-velocity scatter (see Mandel et al. 2017).
    cov : np.ndarray, shape (N, 3, 3)
        Stack of 3x3 measurement covariance matrices in the
        (m_app, c_app, x) basis, one per supernova.

    Attributes
    ----------
    inv_cov : np.ndarray, shape (N, 3, 3)
        Stack of matrix inverses of ``cov``, computed automatically in
        ``__post_init__``.
    num_samples : int
        Number of supernovae in the dataset (length of ``cov``).
    data_params_names : list of str
        Ordered list of scalar-array field names (excludes ``cov`` and
        ``inv_cov``).

    Notes
    -----
    This dataclass is frozen (immutable after construction). Passing
    ``pandas.Series`` objects for any scalar field is safe; they are cast
    to ``numpy.ndarray`` automatically.
    """
    m_app: np.ndarray
    c_app: np.ndarray
    x: np.ndarray
    z: np.ndarray
    sigma_z: np.ndarray
    dist_mod: np.ndarray
    sigma_mu_z2: np.ndarray
    cov: np.ndarray
    inv_cov: np.ndarray = field(init=False)
    num_samples: int = field(init=False)
    data_params_names = ['m_app', 'c_app', 'x', 'z', 'sigma_z', 'dist_mod', 'sigma_mu_z2']

    def __post_init__(self):
        for param in self.data_params_names: # in case pd.Series objects are passed
            object.__setattr__(self, param, np.asarray(getattr(self, param)))
            
        # self.inv_cov = np.linalg.inv(self.cov)
        object.__setattr__(self, 'inv_cov', np.linalg.inv(self.cov))
        object.__setattr__(self, 'num_samples', len(self.cov))
    

    def __repr__(self) -> str:
        data_keys = ", ".join(self.data_params_names) #", ".join(sorted(self.data.keys()))
        cov_shape = getattr(self.cov, "shape", None)
        inv_cov_shape = getattr(self.inv_cov, "shape", None)
        return (
            f"{self.__class__.__name__}(num_samples={self.num_samples}, "
            f"data_keys=[{data_keys}], cov_shape={cov_shape}, inv_cov_shape={inv_cov_shape})"
        )

    def __str__(self) -> str:
        lines = [f"{self.__class__.__name__} with {self.num_samples} samples", "data arrays:"]
        for param in self.data_params_names:
            v = getattr(self, param)
            try:
                arr = np.asarray(v)
                preview = np.array2string(arr.flatten()[:5], separator=", ", max_line_width=80)
                shape = arr.shape
            except Exception:
                preview = "<unavailable>"
                shape = getattr(v, "shape", None)
            lines.append(f"  - {param}: shape={shape}, preview={preview}")
        lines.append(f"cov shape: {getattr(self.cov, 'shape', None)}")
        lines.append(f"inv_cov shape: {getattr(self.inv_cov, 'shape', None)}")
        return "\n".join(lines)
    
    def _repr_html_(self) -> str:
        from html import escape
        title = f"<div><strong>{escape(self.__class__.__name__)} with {self.num_samples} samples</strong></div>"
        style = (
            "<style>"
            "table.simplebayesn{border-collapse:collapse;font-family:Arial,Helvetica,sans-serif;font-size:12px}"
            "table.simplebayesn th, table.simplebayesn td{border:1px solid #ddd;padding:6px;text-align:left;vertical-align:top}"
            "table.simplebayesn th{background:#f7f7f7}"
            "</style>"
        )

        rows = [style, title, "<table class='simplebayesn'><thead><tr><th>key</th><th>shape</th><th>preview (first elements)</th></tr></thead><tbody>"]

        for param in self.data_params_names:
            v = getattr(self, param)
            try:
                arr = np.asarray(v)
                shape = arr.shape
                preview_arr = arr.flatten()[:8]
                preview = escape(np.array2string(preview_arr, separator=", ", max_line_width=300))
            except Exception:
                shape = getattr(v, "shape", None)
                preview = "<i>unavailable</i>"
            rows.append(f"<tr><td>{escape(param)}</td><td>{escape(str(shape))}</td><td><code>{preview}</code></td></tr>")

        rows.append(f"<tr><td><strong>cov</strong></td><td>{escape(str(getattr(self.cov, 'shape', None)))}</td><td></td></tr>")
        rows.append(f"<tr><td><strong>inv_cov</strong></td><td>{escape(str(getattr(self.inv_cov, 'shape', None)))}</td><td></td></tr>")
        rows.append("</tbody></table>")

        return "\n".join(rows)
    
    def __getitem__(self, i):
        return dict(
            m_app = self.m_app[i],
            c_app = self.c_app[i],
            x = self.x[i],
            z = self.z[i],
            sigma_z = self.sigma_z[i],
            dist_mod = self.dist_mod[i],
            sigma_mu_z2 = self.sigma_mu_z2[i],
            cov = self.cov[i],
            inv_cov = self.inv_cov[i]
        )

@dataclass
class GibbsChainData:
    """
    Container for the output of a Gibbs (or emcee) MCMC chain over the
    BayeSN hierarchical model parameters.

    Stores both global (population-level) hyperparameters and per-supernova
    latent parameters at each sampled iteration. All arrays are allocated
    lazily: if ``num_chain_samples`` and ``num_data_samples`` are provided at
    construction, zero-filled arrays are created for any field left as
    ``None``; otherwise fields remain ``None`` until populated (e.g. via
    ``load``).

    Parameters
    ----------
    num_chain_samples : int or None
        Number of MCMC iterations stored. Set automatically after loading.
    num_data_samples : int or None
        Number of supernovae in the dataset. Set automatically after loading.

    Global hyperparameter arrays (each shape ``(num_chain_samples,)``)
    -------------------------------------------------------------------
    tau : np.ndarray or None
        Scale parameter of the dust extinction prior E(B-V) ~ Exp(tau).
    RB : np.ndarray or None
        Total-to-selective dust extinction ratio R_B = A_B / E(B-V).
    x0 : np.ndarray or None
        Population mean stretch parameter.
    sigmax2 : np.ndarray or None
        Population variance of the stretch distribution.
    c0_int : np.ndarray or None
        Population mean intrinsic color.
    alphac_int : np.ndarray or None
        Slope of the intrinsic color-stretch relation.
    sigmac_int2 : np.ndarray or None
        Residual variance of the intrinsic color distribution.
    M0_int : np.ndarray or None
        Population mean intrinsic absolute magnitude.
    alpha : np.ndarray or None
        Intrinsic stretch-magnitude slope (alpha * x).
    beta_int : np.ndarray or None
        Intrinsic color-magnitude slope (beta_int * c_int).
    sigma_int2 : np.ndarray or None
        Residual intrinsic magnitude scatter variance.

    Latent per-supernova arrays (each shape ``(num_chain_samples, num_data_samples)``)
    ----------------------------------------------------------------------------------
    m_app : np.ndarray or None
        Posterior apparent magnitude latent variable for each SN.
    c_app : np.ndarray or None
        Posterior apparent color latent variable for each SN.
    x : np.ndarray or None
        Posterior stretch latent variable for each SN.
    E : np.ndarray or None
        Posterior dust reddening E(B-V) for each SN.
    dist_mod : np.ndarray or None
        Posterior distance modulus for each SN.

    Attributes
    ----------
    sigmax : np.ndarray
        Square root of ``sigmax2`` (population stretch standard deviation).
    sigmac_int : np.ndarray
        Square root of ``sigmac_int2`` (intrinsic color standard deviation).
    sigma_int : np.ndarray
        Square root of ``sigma_int2`` (intrinsic magnitude scatter).
    global_params_names : list of str
        Names of all global hyperparameter fields.
    latent_params_names : list of str
        Names of all per-SN latent parameter fields.
    """
    num_chain_samples: int | None = None
    num_data_samples: int | None = None
    # globals
    tau: np.ndarray | None = None
    RB: np.ndarray | None = None
    x0: np.ndarray | None = None
    sigmax2: np.ndarray | None = None
    c0_int: np.ndarray | None = None
    alphac_int: np.ndarray | None = None
    sigmac_int2: np.ndarray | None = None
    M0_int: np.ndarray | None = None
    alpha: np.ndarray | None = None
    beta_int: np.ndarray | None = None
    sigma_int2: np.ndarray | None = None
    # latents
    m_app: np.ndarray | None = None
    c_app: np.ndarray | None = None
    x: np.ndarray | None = None
    E: np.ndarray | None = None
    dist_mod: np.ndarray | None = None
    global_params_names = ['tau', 'RB', 'x0', 'sigmax2',
                           'c0_int', 'alphac_int', 'sigmac_int2',
                           'M0_int', 'alpha', 'beta_int', 'sigma_int2']
    latent_params_names = ['m_app', 'c_app', 'x', 'E', 'dist_mod']

    def __post_init__(self):
        if self.num_chain_samples is not None and self.num_data_samples is not None:
            for gp in self.global_params_names:
                if getattr(self, gp) is None:
                    setattr(self, gp, np.zeros(self.num_chain_samples, dtype=float))

            for lp in self.latent_params_names:
                if getattr(self, lp) is None:
                    setattr(self, lp, np.zeros((self.num_chain_samples, self.num_data_samples), dtype = float))

    def __setitem__(self, t: int, new_vals: dict[str, np.ndarray]):
        for param in self.global_params_names + self.latent_params_names:
            getattr(self, param)[t] = new_vals[param]

    def __getitem__(self, t):
        if isinstance(t, str):
            return getattr(self, t)
        
        all_latents_exist = all([getattr(self, lp) is not None for lp in self.latent_params_names])
        return {gp: getattr(self, gp)[t] for gp in self.global_params_names} | \
               {lp: getattr(self, lp)[t] if all_latents_exist else None for lp in self.latent_params_names}

    @property
    def sigmax(self):
        return np.sqrt(self.sigmax2)
    
    @property
    def sigmac_int(self):
        return np.sqrt(self.sigmac_int2)
    
    @property
    def sigma_int(self):
        return np.sqrt(self.sigma_int2)

    def load(self, path: str | Path, marginal: bool = False):
        """
        Load chain data from an HDF5 file into this object.

        Supports two formats:

        - **Gibbs format** (default, ``marginal=False``): an HDF5 file written
          by :meth:`save`, containing datasets for every global and latent
          parameter. Both ``num_chain_samples`` and ``num_data_samples`` are
          inferred from the shape of the stretch array ``x``.
        - **emcee format** (``marginal=True``): an emcee ``HDFBackend`` file
          containing only the flattened global-parameter chain (no latent
          variables). ``num_chain_samples`` is inferred from the chain length;
          ``num_data_samples`` is not set.

        Parameters
        ----------
        path : str or Path
            Path to the HDF5 file to read.
        marginal : bool, optional
            If ``True``, treat the file as an emcee backend and read only the
            global (marginalized) hyperparameter chain. Default is ``False``.

        Returns
        -------
        GibbsChainData
            The object itself (mutated in place), to allow chained calls such
            as ``GibbsChainData().load(path)``.

        Raises
        ------
        KeyError
            If ``marginal=False`` but the file does not contain latent-variable
            datasets (i.e. it is an emcee file). Use :func:`load_emcee_data` in
            that case.
        """
        path = Path(path)
        if marginal:
            reader = HDFBackend(path, read_only = True)
            global_params = from_param_array(reader.get_chain(flat = True).T)
            for param in self.global_params_names:
                setattr(self, param, global_params[param])
            self.num_chain_samples = len(global_params['tau'])
            self.num_data_samples = h5py.File(path).attrs.get('num_data_samples', None)
        else:
            with h5py.File(path, 'r') as f:
                for param in self.latent_params_names + self.global_params_names:
                    setattr(self, param, f[param][:])
            self.num_chain_samples, self.num_data_samples = self.x.shape

        return self

    def save(self, path: str | Path):
        """
        Save all chain arrays to an HDF5 file.

        Creates one HDF5 dataset per parameter (both global and latent), using
        the parameter name as the dataset key. The file is created (or
        overwritten) at ``path``.

        Parameters
        ----------
        path : str or Path
            Destination path for the HDF5 file.
        """
        with h5py.File(Path(path), 'w') as f:
            for param in self.latent_params_names + self.global_params_names:
                f.create_dataset(param, data = getattr(self, param))

def load_gibbs_data(path: str | Path):
    """
    Load a full Gibbs chain (globals + latents) from an HDF5 file.

    Convenience wrapper around ``GibbsChainData().load(path)``. The file must
    have been written by :meth:`GibbsChainData.save` and must contain datasets
    for all latent-parameter arrays.

    Parameters
    ----------
    path : str or Path
        Path to the HDF5 file.

    Returns
    -------
    GibbsChainData
        Populated chain object with both global and latent parameter arrays.

    Raises
    ------
    ValueError
        If the file does not contain latent-variable datasets (e.g. it is an
        emcee backend file). Use :func:`load_emcee_data` instead.
    """
    try:
        return GibbsChainData().load(path)
    except KeyError:
        raise ValueError('Please use load_emcee_data to open emcee data (latents are missing from provided file)')

def load_emcee_data(path: str | Path):
    """
    Load a marginalized (globals-only) chain from an emcee HDF5 backend file.

    Convenience wrapper around ``GibbsChainData().load(path, marginal=True)``.
    The file must be a valid ``emcee.backends.HDFBackend`` file. Only global
    hyperparameters are loaded; latent per-SN arrays are not populated.

    Parameters
    ----------
    path : str or Path
        Path to the emcee HDF5 backend file.

    Returns
    -------
    GibbsChainData
        Chain object with global parameter arrays populated and latent arrays
        left as ``None``.
    """
    return GibbsChainData().load(path, marginal=True)
