import simplebayesn
import os
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import pandas as pd

def ztf_fnd_gibbs_table(argv = None):
    parser = ArgumentParser(
        description = 'Compute tables of mean +- std estimates of all parameters from posterior chains for ZTF and Foundation'
    )
    parser.add_argument('-iz', '--input_ztf', type = str, help = 'Path of ZTF GibbsData .h5 (input file)')
    parser.add_argument('-if', '--input_fnd', type = str, help = 'Path of Foundation GibbsData .h5 (input file)')
    parser.add_argument('-o', '--output', type = str, help = 'Path of the folder where to save the output tables')

    args = parser.parse_args(argv)

    gd_ztf_h5_path = Path(args.input_ztf)
    gd_fnd_h5_path = Path(args.input_fnd)
    tables_base_path = Path(args.output)

    if gd_ztf_h5_path.suffix != '.h5':
        raise ValueError('Please provide the full path of the input ZTF .h5 file')

    if gd_fnd_h5_path.suffix != '.h5':
        raise ValueError('Please provide the full path of the input Foundation .h5 file')

    if tables_base_path.suffix != '':
        raise ValueError('Please provide the output folder, not output file path')

    gd_ztf = simplebayesn.load_gibbs_data(gd_ztf_h5_path)
    gd_fnd = simplebayesn.load_gibbs_data(gd_fnd_h5_path)

    params = ['tau', 'RB','x0', 'sigmax2','c0_int', 'alphac_int', 'sigmac_int2', 'M0_int', 'alpha', 'beta_int', 'sigma_int2']

    start_idx = 1000

    for gd, label in zip([gd_ztf, gd_fnd], ['ztf', 'fnd']):
        mean, std = {}, {}
        for p in params:
            mean[p] = np.mean(getattr(gd, p)[start_idx:])
            std[p] = np.std(getattr(gd, p)[start_idx:])

        df = pd.DataFrame([mean, std]).T.rename(columns = {0:'mean', 1:'std'})

        print(label, '\n', df)

        df.to_csv(tables_base_path / Path(f'{label}_posterior_estimates.csv'))

if __name__ == '__main__':
    ztf_fnd_gibbs_table()
