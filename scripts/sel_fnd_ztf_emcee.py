import numpy as np
import simplebayesn
from emcee.backends import HDFBackend
import pandas as pd
from astropy.table import Table

# import jax
# print(jax.devices())
# import jax
# jax.config.update('jax_platform_name', 'cpu')
# print(jax.devices())  # Should show CPU

# import jax
# jax.config.update('jax_log_compiles', True)
# jax.config.update('jax_disable_jit', True)


ztf = pd.read_csv('/home/mgiunta/dati/ztf/ztfsniadr2/tables/snia_data.csv', index_col=[0])
ztf_hq_vl = ztf.loc[(ztf.fitquality_flag == 1) & (ztf.lccoverage_flag == 1) & (ztf.redshift <= 0.06)].dropna()
sd_ztf = simplebayesn.preprocess_data(ztf_hq_vl)

cmin, cmax = ztf_hq_vl['c'].min(), ztf_hq_vl['c'].max()
zmin, zmax = ztf_hq_vl['redshift'].min(), ztf_hq_vl['redshift'].max()

official_fit_cosmo = Table.read('/home/mgiunta/dati/Foundation_DR1/Foundation_DR1.FITRES.TEXT', format='ascii').to_pandas()
official_fit_cosmo.loc[official_fit_cosmo['CID'] == 'AT2016aj', 'CID'] = 'AT2016ajl'
fnd = official_fit_cosmo.loc[(official_fit_cosmo.c <= cmax) & (official_fit_cosmo.c >= cmin) & (official_fit_cosmo.zHEL <= zmax) & (official_fit_cosmo.zHEL >= zmin)]
fnd = fnd.rename(columns = {
    'zHEL':'redshift',
    'zHELERR':'redshift_err',
    'x0ERR':'x0_err',
    'cERR':'c_err',
    'x1ERR':'x1_err',
    'COV_c_x0':'cov_x0_c',
    'COV_x1_x0':'cov_x0_x1',
    'COV_x1_c':'cov_x1_c'
})
sd_fnd = simplebayesn.preprocess_data(fnd)

b = HDFBackend('/home/mgiunta/codice/fnd_ztf_sel.h5')
nw = 25
iv = simplebayesn.initialize.sample_initial_values_uniform(
    sd_fnd.num_samples,
    seed = np.arange(nw),
    marginal = True,
    to_param_array = True
)

if __name__ == '__main__':
    simplebayesn.samplers.emcee_sampler(
        nw,
        1000,
        10000,
        iv,
        simplebayesn.priors.emcee.uniform_marginal_log_prior_invgamma_sigmac_int2,
        sd_fnd,
        selection = True,
        kde_args = {'c_app':sd_ztf.c_app, 'z':sd_ztf.z},
        num_sim_per_sample = 2000,
        backend = b,
        parallel = True
    )
