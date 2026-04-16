import numpy as np
import pandas as pd
from ..utils.data import SaltData
from astropy.cosmology import Planck18
# import astropy.units as u
# FlatLamBdaCDM(H0=72 * u.km / u.s / u.Mpc, Om0=0.27)

def preprocess_data(data: pd.DataFrame, cosmo = Planck18, x0_to_mB_offset: float = 10.635, sigma_pec: float = 300):
    """
    Preprocess supernova light-curve fit results into observed quantities and their measurement
    covariances suitable for downstream cosmological analysis.
 
    This function:
    - Copies the input table-like DataFrame and verifies the presence of required columns.
    - Computes the observed apparent magnitude m_app from the SALT2 flux parameter x0 via
        m_app = -2.5 * log10(x0) + x0_to_mB_offset.
    - Assembles other observed quantities directly from DataFrame columns: color (c_app),
        stretch (x), redshift (z) and redshift uncertainty (sigma_z).
    - Computes the distance modulus for each redshift using the provided cosmology object
        (expects an object with a .distmod(redshift).value API, e.g. an astropy.cosmology instance).
    - Computes an extra redshift-driven distance modulus variance term sigma_mu_z2 that includes
        measurement redshift uncertainty and a contribution from peculiar velocity scatter.
    - Transforms the covariance matrices for (x0, c, x1) into the covariance matrices for
        (m_app, c_app, x) using the analytic Jacobian of the m_app transformation.
 
    Parameters
    ----------
    data : pandas.DataFrame
        Input table containing fitted SALT2 parameters and their measurement uncertainties.
        Required columns (all must be present):
            - 'x0'           : SALT2 flux scale parameter
            - 'c'            : color parameter
            - 'x1'           : stretch parameter
            - 'redshift'     : observed redshift
            - 'redshift_err' : redshift uncertainty (one-sigma)
            - 'x0_err'       : one-sigma uncertainty on x0
            - 'c_err'        : one-sigma uncertainty on c
            - 'x1_err'       : one-sigma uncertainty on x1
            - 'cov_x0_c'     : covariance between x0 and c
            - 'cov_x0_x1'    : covariance between x0 and x1
            - 'cov_x1_c'     : covariance between x1 and c
 
        Alternatively, if the column ``'mB'`` is present it is used directly as the apparent
        magnitude (bypassing the x0 log-transform), and the covariance transformation uses
        ``'mB_err'``, ``'cov_mB_x1'``, and ``'cov_mB_c'`` instead of their x0 counterparts.
    cosmo : object, optional
        Cosmology object used to compute the distance modulus. The object must provide a
        method or function distmod(z) returning a quantity or object with a .value attribute
        (e.g. astropy.cosmology.Cosmology.distmod). Default is Planck18.
    x0_to_mB_offset : float, optional
        Additive offset applied when converting -2.5*log10(x0) to an apparent magnitude scale.
        Default is 10.635.
    sigma_pec : float, optional
        Peculiar velocity scatter in km/s (used to increase error bars on inferred distance
        moduli using linear propagation of errors in redshift). Default is 300 km/s.
 
    Returns
    -------
    SaltData
        A frozen dataclass instance (SaltData) with the following fields:
        - m_app : np.ndarray, shape (N,)
            Apparent magnitude computed from x0 (or taken directly from 'mB').
        - c_app : np.ndarray, shape (N,)
            Observed color (c).
        - x : np.ndarray, shape (N,)
            Observed stretch (x1).
        - z : np.ndarray, shape (N,)
            Redshift.
        - sigma_z : np.ndarray, shape (N,)
            Redshift uncertainty.
        - dist_mod : np.ndarray, shape (N,)
            Distance modulus computed from cosmo for each redshift.
        - sigma_mu_z2 : np.ndarray, shape (N,)
            Variance contribution to distance modulus from redshift errors
            and peculiar velocity scatter.
        - cov : np.ndarray, shape (N, 3, 3)
            Stack of transformed 3x3 covariance matrices for (m_app, c_app, x) for each
            input row. The transformation applies the Jacobian corresponding to
            m_app = -2.5 * log10(x0) + offset while leaving c and x1 unchanged.
        - inv_cov : np.ndarray, shape (N, 3, 3)
            Stack of inverses of the transformed covariance matrices (one 3x3 inverse per row).
 
    Raises
    ------
    ValueError
        If any of the required columns listed above are missing from `data`. Note that x0 values
        must be strictly positive for the logarithm; non-positive x0 will produce NaNs or
        infinities in the magnitude and its covariance.
 
    Notes
    -----
    - The internal covariance transformation assumes the input covariance for each row is given
      in the order (x0, c, x1) with variances provided by x0_err**2, c_err**2, x1_err**2 and
      the off-diagonal covariances given by cov_x0_c, cov_x0_x1, cov_x1_c.
    - The function returns a SaltData frozen dataclass (not a generic dict) for safe, typed
      downstream usage; fields are plain numpy arrays and Python-native containers for ease of use.
 
    Examples
    --------
    ::
 
        import pandas as pd
        from astropy.cosmology import FlatLambdaCDM
        import astropy.units as u
 
        df = pd.read_csv('my_salt2_fits.csv')
        cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3)
        salt_data = preprocess_data(df, cosmo=cosmo)
        print(salt_data)
    """
    
    def compute_transformed_cov(row: pd.Series, sigma_mB: bool):
        """
        Compute the 3x3 measurement covariance matrix in the (m_app, c_app, x) basis
        for a single supernova.
 
        If ``sigma_mB`` is ``False``, the covariance is first assembled in the
        (x0, c, x1) basis and then propagated to (m_app, c, x1) via the Jacobian
        of ``m_app = -2.5 * log10(x0) + offset``. If ``sigma_mB`` is ``True``,
        the (mB, c, x1) covariance is used directly without transformation.
 
        Parameters
        ----------
        row : pandas.Series
            A single row from the input DataFrame, containing the photometric fit
            parameters and their uncertainties.
        sigma_mB : bool
            If ``True``, read covariance entries from the ``mB``-based columns
            (``'mB_err'``, ``'cov_mB_x1'``, ``'cov_mB_c'``) and skip the
            Jacobian transformation. If ``False``, use the ``x0``-based columns
            and apply the log-transform Jacobian.
 
        Returns
        -------
        np.ndarray, shape (3, 3)
            Measurement covariance matrix in the (m_app, c_app, x) basis.
        """
        if not sigma_mB:
            x0 = row['x0']

            sigma_x0  = row['x0_err']
            sigma_x1  = row['x1_err']
            sigma_c   = row['c_err']
            cov_x0_x1 = row['cov_x0_x1']
            cov_x0_c  = row['cov_x0_c']
            cov_x1_c  = row['cov_x1_c']

            cov = np.array([
                [sigma_x0**2, cov_x0_c, cov_x0_x1],
                [cov_x0_c, sigma_c**2, cov_x1_c],
                [cov_x0_x1, cov_x1_c, sigma_x1**2]
            ])
            J = np.eye(3)
            J[0, 0] = -2.5 / (x0 * np.log(10))

            return J @ cov @ J.T
        else:
            sigma_mB  = row['mB_err']
            sigma_x1  = row['x1_err']
            sigma_c   = row['c_err']
            cov_mB_x1 = row['cov_mB_x1']
            cov_mB_c  = row['cov_mB_c']
            cov_x1_c  = row['cov_x1_c']

            cov = np.array([
                [sigma_mB**2, cov_mB_c, cov_mB_x1],
                [cov_mB_c, sigma_c**2, cov_x1_c],
                [cov_mB_x1, cov_x1_c, sigma_x1**2]
            ])

            return cov

    df = data.copy()
    # needed_vars = {
    #     'x0', 'c', 'x1',
    #     'redshift', 'redshift_err',
    #     'x0_err', 'c_err', 'x1_err',
    #     'cov_x0_c', 'cov_x0_x1', 'cov_x1_c'
    # }
    # missing_vars = needed_vars - set(df.columns)
    # if missing_vars:
    #     raise ValueError(f"Missing variables: {', '.join(missing_vars)}")

    if 'mB' in df:
        m_app = df['mB'].to_numpy()
    else:
        m_app = -2.5*np.log10(df['x0'].to_numpy())+x0_to_mB_offset

    observed_data_arr = {
        'm_app':m_app,
        'c_app':df['c'].to_numpy(),
        'x':df['x1'].to_numpy(),
        'z':df['redshift'].to_numpy(),
        'sigma_z':df['redshift_err'].to_numpy(),
        'dist_mod':cosmo.distmod(df['redshift'].to_numpy()).value,
        'sigma_mu_z2': (
            (5 / (df['redshift'].to_numpy()*np.log(10))) ** 2 *
            (df['redshift_err'].to_numpy()**2 + (sigma_pec/299792.458)**2)
        )
    }
    
    observed_data_cov_arr = np.stack(df.apply(compute_transformed_cov, axis=1, args=(('sigma_mB' in df),))) # (N, 3, 3)

    
    return SaltData(**{
        **observed_data_arr,
        'cov':observed_data_cov_arr
    })
