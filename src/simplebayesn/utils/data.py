import numpy as np
from dataclasses import dataclass, field
import h5py
from pathlib import Path
from emcee.backends import HDFBackend
from .param_array import from_param_array

@dataclass(frozen=True)
class SaltData:
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
        object.__setattr__(self, 'data', {
            param:np.asarray(getattr(self, param)) for param in self.data_params_names # in case pd.Series objects are passed
        })
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
        if all([getattr(self, lp) is not None for lp in self.latent_params_names]):
            return {
                'latent_params': {lp: getattr(self, lp)[t] for lp in self.latent_params_names},
                'global_params': {gp: getattr(self, gp)[t] for gp in self.global_params_names}
            }
        else:
            return {
                'latent_params': {lp: None for lp in self.latent_params_names},
                'global_params': {gp: getattr(self, gp)[t] for gp in self.global_params_names}
            }

    def get_samples(self):
        return {
            'latent_params': {lp: getattr(self, lp) for lp in self.latent_params_names},
            'global_params': {gp: getattr(self, gp) for gp in self.global_params_names}
        }

    def load(self, path: str | Path, marginal: bool = False):
        if marginal:
            reader = HDFBackend(Path(path), read_only = True)
            global_params = from_param_array(reader.get_chain(flat = True).T)
            for param in self.global_params_names:
                setattr(self, param, global_params[param])
            self.num_chain_samples = len(global_params['tau'])
        else:
            with h5py.File(Path(path), 'r') as f:
                for param in self.latent_params_names + self.global_params_names:
                    setattr(self, param, f[param][:])
            self.num_chain_samples, self.num_data_samples = self.x.shape

        return self

    def save(self, path: str | Path):
        with h5py.File(Path(path), 'w') as f:
            for param in self.latent_params_names + self.global_params_names:
                f.create_dataset(param, data = getattr(self, param))
