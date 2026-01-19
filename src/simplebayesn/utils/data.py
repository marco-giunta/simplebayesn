import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple

class SaltDataCompact(NamedTuple):
    num_samples: int
    m_app: np.ndarray
    c_app: np.ndarray
    x: np.ndarray
    z: np.ndarray
    sigma_z: np.ndarray
    dist_mod: np.ndarray
    sigma_mu_z2: np.ndarray
    cov: np.ndarray
    inv_cov: np.ndarray

@dataclass(frozen=True)
class SaltData:
    """
    Container for SALT (Spectral Adaptive Lightcurve Template) type Ia supernova data.

    This class is intended to hold the per-supernova arrays and associated covariance
    matrices used in light-curve fitting and cosmological analyses. The primary data
    is stored in the `data` mapping and exposed through convenience properties.

    Expected attributes
    -------------------
    - data : dict[str, np.ndarray]
        A dictionary of 1D numpy arrays of length N (number of SNe). Required keys:
          - 'm_app' : apparent B-band peak magnitude (float array, shape (N,))
          - 'c_app' : observed color (dimensionless, shape (N,))
          - 'x'     : stretch (x1) parameter (dimensionless, shape (N,))
          - 'z'     : redshift in the heliocentric or CMB frame as appropriate (shape (N,))
          - 'sigma_z' : 1-sigma redshift uncertainty (shape (N,))
          - 'dist_mod' : precomputed distance moduli μ(z) evaluated under a chosen cosmology
                         (same units as magnitudes, shape (N,))
          - 'sigma_mu_z2' : squared uncertainty on the observed distance moduli
                            resulting from propagation of `sigma_z` and an assumed
                            peculiar velocity uncertainty (i.e. σ_μ(z)^2, shape (N,))

    - cov : np.ndarray
        The full data covariance matrix (shape (N, 3, 3)) describing correlated uncertainties
        between samples. `num_samples` is inferred from len(cov). This matrix may include
        measurement errors, intrinsic scatter contributions, and any sample-to-sample
        correlations provided by the user.

    Properties
    ----------
    - num_samples -> int
        The number of samples N implied by the covariance matrix (len(cov)).

    - m_app, c_app, x, z, sigma_z, dist_mod, sigma_mu_z2
        Convenience properties returning the corresponding arrays from `data`.

    Notes
    -----
    - The `dist_mod` entries are expected to be computed for a specific cosmological
      model (e.g., LambdaCDM with chosen parameters). Any change of cosmology requires
      recomputing these values.
    - The `sigma_mu_z2` entries must represent the variance (squared uncertainty)
      on the distance modulus coming from propagating redshift uncertainty and adding
      the contribution from assumed peculiar velocity dispersion (converted to magnitude
      uncertainty). This is provided here so downstream code can combine it with other
      error sources consistently.
    - All arrays should have consistent ordering and length N.
    """
    data: dict[str, np.ndarray]
    cov: np.ndarray
    inv_cov: np.ndarray = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'data', {
            k:np.asarray(v) for k, v in self.data.items() # in case pd.Series objects are passed
        })
        # self.inv_cov = np.linalg.inv(self.cov)
        object.__setattr__(self, 'inv_cov', np.linalg.inv(self.cov))

    @property
    def num_samples(self):
        return len(self.cov)
    @property
    def m_app(self):
        return self.data['m_app']
    @property
    def c_app(self):
        return self.data['c_app']
    @property
    def x(self):
        return self.data['x']
    @property
    def z(self):
        return self.data['z']
    @property
    def sigma_z(self):
        return self.data['sigma_z']
    @property
    def dist_mod(self):
        return self.data['dist_mod']
    @property
    def sigma_mu_z2(self):
        return self.data['sigma_mu_z2']
    

    def __repr__(self) -> str:
        data_keys = ", ".join(self.data.keys()) #", ".join(sorted(self.data.keys()))
        cov_shape = getattr(self.cov, "shape", None)
        inv_cov_shape = getattr(self.inv_cov, "shape", None)
        return (
            f"{self.__class__.__name__}(num_samples={self.num_samples}, "
            f"data_keys=[{data_keys}], cov_shape={cov_shape}, inv_cov_shape={inv_cov_shape})"
        )

    def __str__(self) -> str:
        lines = [f"{self.__class__.__name__} with {self.num_samples} samples", "data arrays:"]
        for k, v in self.data.items():
            try:
                arr = np.asarray(v)
                preview = np.array2string(arr.flatten()[:5], separator=", ", max_line_width=80)
                shape = arr.shape
            except Exception:
                preview = "<unavailable>"
                shape = getattr(v, "shape", None)
            lines.append(f"  - {k}: shape={shape}, preview={preview}")
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

        for k, v in self.data.items():
            try:
                arr = np.asarray(v)
                shape = arr.shape
                preview_arr = arr.flatten()[:8]
                preview = escape(np.array2string(preview_arr, separator=", ", max_line_width=300))
            except Exception:
                shape = getattr(v, "shape", None)
                preview = "<i>unavailable</i>"
            rows.append(f"<tr><td>{escape(k)}</td><td>{escape(str(shape))}</td><td><code>{preview}</code></td></tr>")

        rows.append(f"<tr><td><strong>cov</strong></td><td>{escape(str(getattr(self.cov, 'shape', None)))}</td><td></td></tr>")
        rows.append(f"<tr><td><strong>inv_cov</strong></td><td>{escape(str(getattr(self.inv_cov, 'shape', None)))}</td><td></td></tr>")
        rows.append("</tbody></table>")

        return "\n".join(rows)
    
    @property
    def compact(self):
        return SaltDataCompact(
            num_samples = self.num_samples,
            m_app = np.asarray(self.m_app),
            c_app = np.asarray(self.c_app),
            x = np.asarray(self.x),
            z = np.asarray(self.z), 
            sigma_z = np.asarray(self.sigma_z),
            dist_mod = np.asarray(self.dist_mod), 
            sigma_mu_z2 = np.asarray(self.sigma_mu_z2),
            cov = np.asarray(self.cov),
            inv_cov = np.asarray(self.inv_cov)
        )
    
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
    num_chain_samples: int
    num_data_samples: int
    global_params: dict[str, np.ndarray] = field(init=False)
    latent_params: dict[str, np.ndarray] = field(init=False)

    def __post_init__(self):
        num_chain_samples = self.num_chain_samples
        num_data_samples = self.num_data_samples
        self.global_params = {
            'tau': np.zeros(num_chain_samples, dtype=float),
            'RB': np.zeros(num_chain_samples, dtype=float),
            'x0': np.zeros(num_chain_samples, dtype=float),
            'sigmax2': np.zeros(num_chain_samples, dtype=float),
            'c0_int': np.zeros(num_chain_samples, dtype=float),
            'alphac_int': np.zeros(num_chain_samples, dtype=float),
            'sigmac_int2': np.zeros(num_chain_samples, dtype=float),
            'M0_int': np.zeros(num_chain_samples, dtype=float),
            'alpha': np.zeros(num_chain_samples, dtype=float),
            'beta_int': np.zeros(num_chain_samples, dtype=float),
            'sigma_int2': np.zeros(num_chain_samples, dtype=float),
        }
        self.latent_params = {
            'm_app': np.zeros((num_chain_samples, num_data_samples), dtype = float),
            'c_app': np.zeros((num_chain_samples, num_data_samples), dtype = float),
            'x': np.zeros((num_chain_samples, num_data_samples), dtype = float),
            'E': np.zeros((num_chain_samples, num_data_samples), dtype = float),
            'dist_mod': np.zeros((num_chain_samples, num_data_samples), dtype = float)
        }

    def set_latent(self, t: int, new_vals: dict[str, np.ndarray]):
        for key in new_vals.keys():
            self.latent_params[key][t] = new_vals[key]
    def set_global(self, t: int, new_vals: dict[str, np.ndarray]):
        for key in new_vals.keys():
            self.global_params[key][t] = new_vals[key]

    def __getitem__(self, t):
        return {
            'latent_params': {k:self.latent_params[k][t] for k in self.latent_params.keys()},
            'global_params': {k:self.global_params[k][t] for k in self.global_params.keys()},
        }

    def get_samples(self):
        return {
            'latent_params': self.latent_params,
            'global_params': self.global_params
        }

    @property
    def m_app(self):
        return self.latent_params['m_app']
    @property
    def c_app(self):
        return self.latent_params['c_app']
    @property
    def x(self):
        return self.latent_params['x']
    @property
    def E(self):
        return self.latent_params['E']
    @property
    def dist_mod(self):
        return self.latent_params['dist_mod']
    
    @property
    def tau(self):
        return self.global_params['tau']
    @property
    def RB(self):
        return self.global_params['RB']
    @property
    def x0(self):
        return self.global_params['x0']
    @property
    def sigmax2(self):
        return self.global_params['sigmax2']
    @property
    def c0_int(self):
        return self.global_params['c0_int']
    @property
    def alphac_int(self):
        return self.global_params['alphac_int']
    @property
    def sigmac_int2(self):
        return self.global_params['sigmac_int2']
    @property
    def M0_int(self):
        return self.global_params['M0_int']
    @property
    def alpha(self):
        return self.global_params['alpha']
    @property
    def beta_int(self):
        return self.global_params['beta_int']
    @property
    def sigma_int2(self):
        return self.global_params['sigma_int2']

class GibbsChainDataCompact(NamedTuple):
    M0_int: np.ndarray
    alpha: np.ndarray
    beta_int: np.ndarray
    sigma_int2: np.ndarray
    c0_int: np.ndarray
    alphac_int: np.ndarray
    sigmac_int2: np.ndarray
    x0: np.ndarray
    sigmax2: np.ndarray
    tau: np.ndarray
    RB: np.ndarray
    m_app: np.ndarray
    c_app: np.ndarray
    x: np.ndarray
    E: np.ndarray
    dist_mod: np.ndarray

    def __getitem__(self, t):
        return {
            'global_params': {
                'M0_int':self.M0_int[t],
                'alpha':self.alpha[t],
                'beta_int':self.beta_int[t],
                'sigma_int2':self.sigma_int2[t],
                'c0_int':self.c0_int[t],
                'alphac_int':self.alphac_int[t],
                'sigmac_int2':self.sigmac_int2[t],
                'x0':self.x0[t],
                'sigmax2':self.sigmax2[t],
                'tau':self.tau[t],
                'RB':self.RB[t]
            },
            'latent_params': {
                'm_app':self.m_app[t],
                'c_app':self.c_app[t],
                'x':self.x[t],
                'E':self.E[t],
                'dist_mod':self.dist_mod[t]
            }
        }
    def get_samples(self):
        return {
            'global_params': {
                'M0_int':self.M0_int,
                'alpha':self.alpha,
                'beta_int':self.beta_int,
                'sigma_int2':self.sigma_int2,
                'c0_int':self.c0_int,
                'alphac_int':self.alphac_int,
                'sigmac_int2':self.sigmac_int2,
                'x0':self.x0,
                'sigmax2':self.sigmax2,
                'tau':self.tau,
                'RB':self.RB
            },
            'latent_params': {
                'm_app':self.m_app,
                'c_app':self.c_app,
                'x':self.x,
                'E':self.E,
                'dist_mod':self.dist_mod
            }
        }
    @property
    def num_chain_samples(self):
        return len(self.tau)
