import numpy as np
from ..utils.data import SaltData
from astropy.cosmology import Planck18

def simulate_simplebayesn_salt_data_from_ztfhqvl_fit(num_samples: int,
            global_params_true: dict = None,
            seed: int = None,
            sigma_pec: float = 300,
            cosmo = Planck18,
            c_app_obs_lim: tuple[float] = None,
            x_obs_lim: tuple[float] = None):
    """
    Simulate SALT-like supernova data using a forward SimpleBayeSN-style generative model
    and empirical predictors derived from a high-quality, volume-limited ZTF fit.
    This function generates latent and observed SALT parameters for a population of
    simulated Type Ia supernovae. Latent parameters (intrinsic stretch x, intrinsic
    color c_int, intrinsic absolute magnitude M_int, and an exponential extrinsic
    reddening E) are drawn from the SimpleBayeSN global priors conditioned on parameters
    (supplied via `global_params_true` or taken from get_sbsn2017_global_params()).
    Observed quantities (apparent magnitude, observed color, observed stretch and
    redshift) are produced by adding measurement-like noise and simple empirical
    predictors fit to the high-quality volume-limited ZTF sample.
    
    Forward model
    -------------
    - Intrinsic latent sampling:
        - x ~ Normal(x0, sqrt(sigmax2))
        - c_int ~ Normal(c0_int + alphac_int * x, sqrt(sigmac_int2))
        - M_int ~ Normal(M0_int + alpha * x + beta_int * c_int, sqrt(sigma_int2))
    - Extrinsic (dust-like) variable:
        - E ~ Exponential(tau)
        - M_ext = M_int + RB * E
        - c_app = c_int + E
    - Redshift sampling:
        - z is sampled from a simple linearly-varying PDF over [0.015, 0.06].
            The PDF is specified to increase linearly across the support (empirically
            fit to the ZTF high-quality volume-limited sample) and sampled via inverse
            transform sampling.
        - sigma_z is set to the median ZTF redshift uncertainty (median_sigma_z),
            tiled per sample because the empirical distribution is sharply peaked.
    - Distance modulus and observational dispersion:
        - The cosmological distance modulus dist_mod_of_z = cosmo.distmod(z)
        - Observational scatter on distance modulus is constructed from the
            quadrature of median redshift uncertainty and a peculiar-velocity term
            (sigma_pec), converted to magnitudes: s = sqrt(sigma_z^2 + sigma_pec^2) * 5/(z*ln(10))
        - The observed distance modulus is drawn as dist_mod = dist_mod_of_z + s * Normal(0,1)
    - Observed SALT vectors:
        - Observed (m_app_obs, c_app_obs, x_obs) are formed by adding correlated
            multivariate Gaussian measurement noise with the median ZTF SALT covariance
            matrix (median_ztf_hq_vl_cov).
    - Selection / cuts:
        - Optional box cuts on observed color and stretch can be applied via
            c_app_obs_lim and x_obs_lim as (min, max) tuples. Only objects that satisfy
            both cuts are retained in the returned arrays.
    
    Parameters
    ----------
    - num_samples : int
            Number of simulated events to generate (before applying optional selection cuts).
    - global_params_true : dict or None, optional
            Dictionary of global SimpleBayeSN-style parameters (e.g. x0, sigmax2, c0_int,
            alphac_int, sigmac_int2, M0_int, alpha, beta_int, sigma_int2, tau, RB, etc.).
            If None, defaults are obtained from get_sbsn2017_global_params().
    - seed : int or None, optional
            Seed for the random number generator. If None, draws are non-deterministic.
    - sigma_pec : float, optional (default=300)
            Peculiar velocity (km/s) used to compute the contribution to distance
            modulus uncertainty (converted to magnitudes).
    - cosmo : object, optional
            Cosmology object with a .distmod(z) method (e.g. astropy Planck18). Used to
            compute the noiseless distance modulus at redshift z.
    - c_app_obs_lim : tuple(float, float) or None, optional
            Optional (min, max) box cut applied to observed color c_app_obs.
            If None, no color cut is applied.
    - x_obs_lim : tuple(float, float) or None, optional
            Optional (min, max) box cut applied to observed stretch x_obs.
            If None, no stretch cut is applied.

    Returns
    -------
    A dict with keys:
    - observed_data : SaltData
            A SaltData-like container constructed from the simulated observed quantities.
            The contained data dict includes arrays (already filtered by any applied
            cuts) for:
                - m_app : observed apparent magnitude (array)
                - c_app : observed color (array)
                - x     : observed stretch (array)
                - z     : redshift (array)
                - sigma_z : per-object redshift uncertainty (array)
                - dist_mod : cosmological distance modulus at z (dist_mod_of_z, array)
                - sigma_mu_z2 : variance on the distance modulus from redshift & peculiar velocity (s^2, array)
            The SaltData also receives:
                - cov : an array of shape (num_samples, 3, 3) equal to the median ZTF SALT
                                covariance matrix tiled for each generated sample (the tiling is
                                performed with length equal to the original num_samples).
                - inv_cov : the inverse of cov, similarly tiled.
            Note: cov/inv_cov are constructed by tiling the median covariance for the
            full generated sample size (num_samples) before any selection; observed
            arrays inside data are filtered by selection cuts.
    - latent_params_true : dict
            Dictionary of latent (true) parameters corresponding to the retained
            simulated objects (after selection):
                - m_app : noiseless apparent magnitude before measurement noise (array)
                - c_app : noiseless apparent color (c_int + E) (array)
                - x     : intrinsic stretch x (array)
                - E     : extrinsic reddening/dust E (array)
                - dist_mod : noisy distance modulus drawn as dist_mod_of_z + s*Normal(0,1) (array)
    - global_params_true : dict
            The global parameters used to generate the sample (either the provided
            dictionary or the defaults from get_sbsn2017_global_params()).
    Notes / usage
    - This routine is intended for forward simulations that combine a SimpleBayeSN
        style generative model for intrinsic+extrinsic supernova parameters with
        empirical, lightweight predictors for the quantities that the model is
        conditioned on (redshift distribution, median measurement covariances, etc.).
    - The redshift distribution and measurement covariance choices are intentionally
        simple approximations derived from the high-quality ZTF volume-limited fit:
        redshift is sampled from a linearly-varying PDF over a fixed support, and
        covariances & sigma_z are set to their median empirical values.
    """
    def sample_linear_pdf(x0, x1, y0, y1, size=1, random_state=None):
        rng = np.random.default_rng(random_state) if not isinstance(random_state, np.random._generator.Generator) else random_state
        # Linearly defined PDF: p(x) = a*x + b
        # Find slope a and intercept b from given values
        a = (y1 - y0) / (x1 - x0)
        b = y0 - a * x0

        # Normalize PDF
        # Integral over [x0,x1] = (a/2)*(x1^2 - x0^2) + b*(x1 - x0)
        integral = (a * (x1**2 - x0**2)) / 2 + b * (x1 - x0)

        # Normalized PDF: p(x) = (a*x + b) / integral

        # CDF: integrate normalized PDF:
        # CDF(x) = (a/2*x^2 + b*x)/integral - constant to adjust CDF(x0)=0
        def cdf(x):
            return ((a * x**2 / 2 + b * x) - (a * x0**2 / 2 + b * x0)) / integral

        # Inverse CDF via solving quadratic for x given u in [0,1]:
        # u = CDF(x)
        # Let C = a/2, D = b, E = -u*integral - (C*x0^2 + D*x0)
        # Solve quadratic C*x^2 + D*x + E = 0
        def inv_cdf(u):
            C = a / 2
            D = b
            E = -u * integral - (C * x0**2 + D * x0)
            # Quadratic formula
            discriminant = D**2 - 4*C*E
            x_pos = (-D + np.sqrt(discriminant)) / (2*C)
            x_neg = (-D - np.sqrt(discriminant)) / (2*C)
            # Select the root within [x0, x1]
            if x0 <= x_pos <= x1:
                return x_pos
            else:
                return x_neg

        # Generate uniform samples and apply inverse CDF
        u_samples = rng.uniform(0, 1, size=size)
        samples = np.array([inv_cdf(u) for u in u_samples])
        return samples
    
    median_ztf_hq_vl_cov = np.array([
        [0.00146294, 0.00104084, 0.00101442],
        [0.00104084, 0.00103433, 0.00012715],
        [0.00101442, 0.00012715, 0.02105116]
    ])
    median_sigma_z = 1.669e-05

    if global_params_true is None:
        global_params_true = get_sbsn2017_global_params()

    rng = np.random.default_rng(seed)
    # intrinsic latent parameters
    x = rng.normal(global_params_true['x0'], np.sqrt(global_params_true['sigmax2']), num_samples)
    c_int = rng.normal(global_params_true['c0_int'] + global_params_true['alphac_int']*x, np.sqrt(global_params_true['sigmac_int2']))
    M_int = rng.normal(global_params_true['M0_int'] + global_params_true['alpha']*x + global_params_true['beta_int']*c_int, np.sqrt(global_params_true['sigma_int2']))

    # extrinsic latent parameters
    E = rng.exponential(global_params_true['tau'], size = num_samples)
    M_ext = M_int + global_params_true['RB']*E
    c_app = c_int + E

    # distance modulus and redshift
    z = sample_linear_pdf(x0 = 0.015, x1 = 0.06, y0 = 7, y1 = 37, size = num_samples, random_state = rng)
    sigma_z = np.tile(median_sigma_z, num_samples)
    sigma_pec_normalized = sigma_pec / 299792.458
    s = np.sqrt(sigma_z**2 + sigma_pec_normalized**2)*5/(z*np.log(10))
    dist_mod_of_z = cosmo.distmod(z).value
    dist_mod = dist_mod_of_z + s * rng.normal(size = num_samples)
    m_app = M_ext + dist_mod

    m_app_obs, c_app_obs, x_obs = (np.column_stack([m_app, c_app, x]) + rng.multivariate_normal([0, 0, 0], median_ztf_hq_vl_cov, num_samples)).T

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
    num_samples = sum(idx)

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
            cov = np.tile(median_ztf_hq_vl_cov, (num_samples, 1, 1)),
            # inv_cov = np.tile(np.linalg.inv(median_ztf_hq_vl_cov), (num_samples, 1, 1))
        )),
        latent_params_true = dict(
            m_app = m_app[idx],
            c_app = c_app[idx], 
            x = x[idx],
            E = E[idx],
            dist_mod = dist_mod[idx],
        ),
        global_params_true = global_params_true,
    )

def simulate_simplebayesn_salt_data_from_ztf_salt_data(observed_data: SaltData,
                                                       global_params_true: dict,
                                                       seed: int = None,
                                                       c_app_obs_lim: tuple[float] = None,
                                                       x_obs_lim: tuple[float] = None):
    return
