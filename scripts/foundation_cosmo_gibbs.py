import pandas as pd
from astropy.table import Table
import simplebayesn
import os
from argparse import ArgumentParser
from pathlib import Path

def foundation_cosmo_gibbs(argv = None):
    parser = ArgumentParser(
        description = 'Fit SimpleBayeSN Gibbs sampler on Foundation DR1 data (official fits of the cosmo sample only)'
    )

    parser.add_argument('-i', '--input', type = str, help = 'Path of the Foundation DR1 cosmo .FITRES.TEXT file')
    parser.add_argument('-o', '--output', type = str, help = 'Path of the GibbsData .h5 MCMC output file')

    args = parser.parse_args(argv)

    fnd_fitres_path = Path(args.input)
    gd_fnd_cosmo_h5_path = Path(args.output)

    if fnd_fitres_path.suffix != '.TEXT':
        raise ValueError('Please provide the input Foundatiomn .FITRES.TEXT file')
    
    if gd_fnd_cosmo_h5_path.suffix != '.h5':
        raise ValueError('Please provide the full path of the output .h5 file')

    os.makedirs(gd_fnd_cosmo_h5_path.parent, exist_ok = True)

    fnd_cosmo = Table.read(fnd_fitres_path, format = 'ascii').to_pandas()
    fnd_cosmo = fnd_cosmo.rename(columns = {
        'zHEL':      'redshift',
        'zHELERR':   'redshift_err',
        'mBERR':     'mb_err',
        'x1ERR':     'x1_err',
        'cERR':      'c_err',
        'x0ERR':     'x0_err',
        'COV_x1_x0': 'cov_x0_x1',
        'COV_c_x0':  'cov_x0_c',
        'COV_x1_c':  'cov_x1_c'
    })
    fnd_cosmo = fnd_cosmo.loc[fnd_cosmo['c'] >= -0.2]
    print(f'number of SNe: {len(fnd_cosmo)}')

    sd_fnd_cosmo = simplebayesn.preprocess_data(fnd_cosmo)

    prior_params = simplebayesn.priors.gibbs.get_priors_params_uniform_invgamma()
    iv = simplebayesn.initialize.sample_initial_values_uniform(num_samples = sd_fnd_cosmo.num_samples, seed = 1234)
    gd_fnd_cosmo = simplebayesn.samplers.gibbs_sampler(iv, prior_params, sd_fnd_cosmo, num_iter = int(1e5), seed = 1234)

    gd_fnd_cosmo.save(gd_fnd_cosmo_h5_path)

    return gd_fnd_cosmo

if __name__ == '__main__':
    foundation_cosmo_gibbs()
