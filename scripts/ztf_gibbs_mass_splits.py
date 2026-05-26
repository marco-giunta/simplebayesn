import pandas as pd
import simplebayesn
import os
from pathlib import Path
from argparse import ArgumentParser

def ztf_gibbs_subset(ztf_subset: pd.DataFrame):
    # convert to simplebayesn SaltData representation
    sd_ztf = simplebayesn.preprocess_data(ztf_subset)
    # uniform prior for slope/mean-like parameters, invgamma(0.003, 0.003) prior for positive variance-like parameters
    prior_params = simplebayesn.priors.gibbs.get_priors_params_uniform_invgamma()
    # initial conditions
    iv = simplebayesn.initialize.sample_initial_values_uniform(num_samples = sd_ztf.num_samples, seed = 1234)
    # generate ZTF GibbsData object
    gd_ztf = simplebayesn.samplers.gibbs_sampler(iv, prior_params, sd_ztf, num_iter = int(1e5), seed = 1234)
    return gd_ztf

def ztf_gibbs_mass_split(argv = None):
    parser = ArgumentParser(
        description = 'Fit SimpleBayeSN Gibbs sampler on subsets of ZTF HQ VL data, partitioned by the available environmental proxies'
    )
    parser.add_argument('-id', '--input_data', type = str, help = 'Path of the main ZTF .csv (input file)')
    parser.add_argument('-ig', '--input_global', type = str, help = 'Path of the globalhost ZTF .csv (input file)')
    parser.add_argument('-il', '--input_local', type = str, help = 'Path of the localhost ZTF .csv (input file)')
    parser.add_argument('-o', '--output', type = str, help = 'Path of the folder where to save the output .h5 files')

    args = parser.parse_args(argv)

    ztf_csv_path = Path(args.input_data)
    ztf_global_csv_path = Path(args.input_global)
    ztf_local_csv_path = Path(args.input_local)
    output_path = Path(args.output)

    if ztf_csv_path.suffix != '.csv':
        raise ValueError('Please provide the input ZTF data .csv file')

    if ztf_global_csv_path.suffix != '.csv':
        raise ValueError('Please provide the input ZTF globalhost data .csv file')

    if ztf_local_csv_path.suffix != '.csv':
        raise ValueError('Please provide the input ZTF localhost data .csv file')

    if output_path.suffix != '':
        raise ValueError('Please provide the path of the folder where to save the output .h5 files')

    os.makedirs(output_path, exist_ok = True)

    ztf = pd.read_csv(ztf_csv_path, index_col=[0])
    ztf_hq_vl = ztf.loc[
        (ztf['fitquality_flag'] == 1) & (ztf['lccoverage_flag'] == 1) & # high quality SNe
        (ztf['redshift'] <= 0.06) & (ztf['redshift'] >= 0.015) & # volume limited & no missing SALT fit
        (ztf['sub_type'].isin([ # nonpeculiar SNe
            'norm', 'norm/99aa', '99aa',
            'norm/04gs', '91t', '99aa/91t'
        ]))
    ]

    globalhost = pd.read_csv(ztf_global_csv_path)
    globalhost = globalhost.loc[globalhost.ztfname.isin(ztf_hq_vl.ztfname)]

    localhost = pd.read_csv(ztf_local_csv_path)
    localhost = localhost.loc[localhost.ztfname.isin(ztf_hq_vl.ztfname)]

    global_vars = {
        'mass': 10,
        'd_dlr': 1.5,
        'restframe_gz': 1,
    }

    local_vars = {
        'mass': 8.9,
        'restframe_gz': 1
    }

    print(f'Fits needed: {(len(global_vars) + len(local_vars))*2}')

    for k, v in global_vars.items():
        print(f'Fitting ZTF HQ VL, globalhost {k} < {v}...')
        ztf_gibbs_subset(
            ztf_hq_vl.loc[ztf_hq_vl.ztfname.isin(
                globalhost.loc[globalhost[k] < v, 'ztfname']
            )]
        ).save(output_path / Path(f'ztf_global_{k}_less_than_{v}.h5'))

        print(f'Fitting ZTF HQ VL, globalhost {k} > {v}...')
        ztf_gibbs_subset(
            ztf_hq_vl.loc[ztf_hq_vl.ztfname.isin(
                globalhost.loc[globalhost[k] > v, 'ztfname']
            )]
        ).save(output_path / Path(f'ztf_global_{k}_larger_than_{v}.h5'))

    for k, v in local_vars.items():
        print(f'Fitting ZTF HQ VL, localhost {k} < {v}...')
        ztf_gibbs_subset(
            ztf_hq_vl.loc[ztf_hq_vl.ztfname.isin(
                localhost.loc[localhost[k] < v, 'ztfname']
            )]
        ).save(output_path / Path(f'ztf_local_{k}_less_than_{v}.h5'))

        print(f'Fitting ZTF HQ VL, localhost {k} > {v}...')
        ztf_gibbs_subset(
            ztf_hq_vl.loc[ztf_hq_vl.ztfname.isin(
                localhost.loc[localhost[k] > v, 'ztfname']
            )]
        ).save(output_path / Path(f'ztf_local_{k}_larger_than_{v}.h5'))

if __name__ == '__main__':
    ztf_gibbs_mass_split()
