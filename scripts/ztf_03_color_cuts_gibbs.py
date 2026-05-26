import pandas as pd
import simplebayesn
import os
from pathlib import Path
from argparse import ArgumentParser

def ztf_gibbs(argv = None):
    parser = ArgumentParser(
        description = 'Fit SimpleBayeSN Gibbs sampler on ZTF HQ VL data with c<=0.3 color cuts'
    )
    parser.add_argument('-i', '--input', type = str, help = 'Path of ZTF .csv (input file)')
    parser.add_argument('-o', '--output', type = str, help = 'Path of GibbsData .h5 MCMC output file')

    args = parser.parse_args(argv)

    ztf_csv_path = Path(args.input)
    gd_ztf_h5_path = Path(args.output)

    if ztf_csv_path.suffix != '.csv':
        raise ValueError('Please provide an input .csv file')

    if gd_ztf_h5_path.suffix != '.h5':
        raise ValueError('Please provide the full path of the output .h5 file')

    os.makedirs(gd_ztf_h5_path.parent, exist_ok = True)

    ztf = pd.read_csv(ztf_csv_path, index_col=[0])
    # SNe with z<=0.015 have NaNs in the x0 etc. SALT fit columns.
    # To discard them, it's equivalent to do .dropna(subset=['x0', 'x1', 'c', ...]) or filter by z>=0.015
    ztf = pd.read_csv(ztf_csv_path, index_col=[0])
    ztf_hq_vl = ztf.loc[
        (ztf['fitquality_flag'] == 1) & (ztf['lccoverage_flag'] == 1) & # high quality SNe
        (ztf['redshift'] <= 0.06) & (ztf['redshift'] >= 0.015) & # volume limited & no missing SALT fit
        (ztf['sub_type'].isin([ # nonpeculiar SNe
            'norm', 'norm/99aa', '99aa',
            'norm/04gs', '91t', '99aa/91t'
        ]))
    ]
    # convert to simplebayesn SaltData representation
    sd_ztf = simplebayesn.preprocess_data(
        ztf_hq_vl.loc[ztf_hq_vl.c <= 0.3]
    )
    # uniform prior for slope/mean-like parameters, invgamma(0.003, 0.003) prior for positive variance-like parameters
    prior_params = simplebayesn.priors.gibbs.get_priors_params_uniform_invgamma()
    # initial conditions
    iv = simplebayesn.initialize.sample_initial_values_uniform(num_samples = sd_ztf.num_samples, seed = 1234)
    # generate ZTF GibbsData object
    gd_ztf = simplebayesn.samplers.gibbs_sampler(iv, prior_params, sd_ztf, num_iter = int(1e5), seed = 1234)

    # save
    gd_ztf.save(gd_ztf_h5_path)

    return gd_ztf

if __name__ == '__main__':
    ztf_gibbs()
