import pandas as pd
import simplebayesn
import os
from pathlib import Path
from argparse import ArgumentParser

def ztf_gibbs(argv = None):
    parser = ArgumentParser(
        description = 'Fit SimpleBayeSN Gibbs sampler on ZTF HQ VL data'
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
    # high quality (flags == True) volume limited (z <= 0.06) subset
    ztf_hq_vl = ztf.loc[(ztf.fitquality_flag == 1) & (ztf.lccoverage_flag == 1) & (ztf.redshift <= 0.06)]
    # discard SNe with missing data (mostly classification)
    ztf_hq_vl = ztf_hq_vl.dropna()
    # convert to simplebayesn SaltData representation
    sd_ztf = simplebayesn.preprocess_data(ztf_hq_vl)
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
