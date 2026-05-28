import simplebayesn
import os
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import pandas as pd

PARAMS_LATEX_MAP = {
    'M0_int': r'$M_0^{\rm{int}}$',
    'alpha': r'$\alpha$',
    'beta_int': r'$\beta_{\rm{int}}$',
    'c0_int': r'$c_0^{\rm{int}}$',
    'alphac_int': r'$\alpha_c^{\rm{int}}$',
    'x0': r'$x_0$',
    'sigma_int2': r'$\sigma_{\rm{int}}^2$',
    'sigmac_int2': r'$\sigma_{c, \rm{int}}^2$',
    'sigmax2': r'$\sigma_x^2$',
    'RB': r'$R_B$',
    'tau': r'$\tau$',
    'sigma_int': r'$\sigma_{\rm{int}}$',
    'sigmac_int': r'$\sigma_{c, \rm{int}}$',
    'sigmax': r'$\sigma_x$',
}
DECIMALS = 3

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
    var_params = ['sigmax2', 'sigmac_int2', 'sigma_int2']

    start_idx = 1000

    for gd, label in zip([gd_ztf, gd_fnd], ['ztf', 'fnd']):
        mean, std = {}, {}
        for p in params:
            mean[p] = np.mean(getattr(gd, p)[start_idx:])
            std[p] = np.std(getattr(gd, p)[start_idx:])

        for p in var_params:
            mean[p.rstrip('2')] = np.mean(np.sqrt(getattr(gd, p)[start_idx:]))
            std[p.rstrip('2')] = np.std(np.sqrt(getattr(gd, p)[start_idx:]))

        df = pd.DataFrame([mean, std]).T.rename(columns = {0:'mean', 1:'std'})

        print(label, '\n', df)

        df.to_csv(tables_base_path / Path(f'{label}_posterior_estimates.csv'))

        df['var_latex'] = df.index.map(PARAMS_LATEX_MAP)
        df['value'] = df.apply(
            lambda row: f"${row['mean']:.{DECIMALS}f} \\pm {row['std']:.{DECIMALS}f}$",
            axis=1
        )

        latex_table = df[['var_latex', 'value']].to_latex(
            index=False,
            header=False,
            escape=False
        )

        print(latex_table)
        with open(tables_base_path / Path(f'{label}_latex_table.txt'), 'w') as f:
            f.write(latex_table)


if __name__ == '__main__':
    ztf_fnd_gibbs_table()
