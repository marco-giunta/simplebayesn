import numpy as np
from ..utils.data import SaltData
from astropy.cosmology import Planck18

def get_sbsn2017_global_params():
    return {
        'M0_int':-19.4,
        'alpha':-0.15,
        'beta_int':2.2,
        'sigma_int2':0.01,
        'c0_int':-0.06,
        'alphac_int':-0.008,
        'sigmac_int2':0.0036,
        'x0':-0.4,
        'sigmax2':1.44,
        'tau':0.07,
        'RB':4.1,
    }

def simulate_simplebayesn_salt_data_from_redshift_saltcov_dist(
        redshift: np.ndarray,
        redshift_err: np.ndarray,
        cov: np.ndarray,
        global_params_true: dict[str, float],
        seed: int = None,
        sigma_pec: float = 300,
        cosmo = Planck18,
        c_app_obs_lim: tuple[float] = None,
        x_obs_lim: tuple[float] = None,
):
    needed_vars = set(get_sbsn2017_global_params().keys())
    missing_vars = needed_vars - set(global_params_true.keys())
    if missing_vars:
        raise ValueError(f"Missing variables: {', '.join(missing_vars)}")

    rng = np.random.default_rng(seed)
    if redshift.shape[0] != cov.shape[0]:
        raise ValueError(f'redshift length {redshift.shape[0]} does not match cov length {cov.shape[0]}')
    if redshift.shape[0] != redshift_err.shape[0]:
        raise ValueError(f'redshift length {redshift.shape[0]} does not match redshift_err length {redshift_err.shape[0]}')
    
    num_samples = redshift.shape[0]

    x = rng.normal(global_params_true['x0'], np.sqrt(global_params_true['sigmax2']), num_samples)
    c_int = rng.normal(global_params_true['c0_int'] + global_params_true['alphac_int']*x, np.sqrt(global_params_true['sigmac_int2']))
    M_int = rng.normal(global_params_true['M0_int'] + global_params_true['alpha']*x + global_params_true['beta_int']*c_int, np.sqrt(global_params_true['sigma_int2']))
    
    E = rng.exponential(global_params_true['tau'], size = num_samples)
    M_ext = M_int + global_params_true['RB']*E
    c_app = c_int + E

    z = redshift
    sigma_z = redshift_err

    sigma_pec_normalized = sigma_pec / 299792.458
    s = np.sqrt(sigma_z**2 + sigma_pec_normalized**2)*5/(z*np.log(10))
    dist_mod_of_z = cosmo.distmod(z).value
    dist_mod = dist_mod_of_z + s * rng.normal(size = num_samples)
    m_app = M_ext + dist_mod

    m_app_obs, c_app_obs, x_obs = (np.column_stack([m_app, c_app, x]) + np.einsum(
        'nij,nj->ni',
        np.linalg.cholesky(cov),
        rng.normal(size = (num_samples, 3))
    )).T

    if c_app_obs_lim is not None:
        idx_c  = c_app_obs <= c_app_obs_lim[1]
        idx_c &= c_app_obs >= c_app_obs_lim[0]
    else:
        idx_c = np.array([True] * num_samples)

    if x_obs_lim is not None:
        idx_x  = x_obs <= x_obs_lim[1]
        idx_x &= x_obs >= x_obs_lim[0]
    else:
        idx_x = np.array([True] * num_samples)

    idx = idx_c & idx_x

    return dict(
        observed_data = SaltData(**dict(
            data = dict(
                m_app = m_app_obs[idx],
                c_app = c_app_obs[idx],
                x = x_obs[idx],
                z = z[idx],
                sigma_z = sigma_z[idx],
                dist_mod = dist_mod_of_z[idx],
                sigma_mu_z2 = (s**2)[idx],
            ),
            cov = cov[idx],
        )),
        latent_params_true = dict(
            m_app = m_app[idx],
            c_app = c_app[idx],
            x = x[idx],
            E = E[idx],
            dist_mod = dist_mod[idx]
        ),
        global_params_true = global_params_true
    )