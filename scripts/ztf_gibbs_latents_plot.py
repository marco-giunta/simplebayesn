from simplebayesn import load_gibbs_data
from simplebayesn.visualize import plot_latent_bias
import os
import numpy as np
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser

def ztf_gibbs_latent_plot(argv = None):
    parser = ArgumentParser(
        description = 'Plot latent variables of Gibbs MCMC chain of ZTF HQ VL data, according to multiple splits'
    )
    parser.add_argument('-iz', '--input_ztf', type = str, help = 'Path of the ZTF data .csv file')
    parser.add_argument('-ih', '--input_h5', type = str, help = 'Path of the GibbsData .h5 (input file)')
    parser.add_argument('-ig', '--input_global', type = str, help = 'Path of the ZTF globalhost_data.csv file')
    parser.add_argument('-il', '--input_local', type = str, help = 'Path of the ZTF localhost_data.csv file')
    parser.add_argument('-o', '--output', type = str, help = 'Path of the folder where to save the output plots')

    args = parser.parse_args(argv)

    ztf_csv_path = Path(args.input_ztf)
    gd_ztf_h5_path = Path(args.input_h5)
    global_path = Path(args.input_global)
    local_path = Path(args.input_local)
    fig_path = Path(args.output)

    if ztf_csv_path.suffix != '.csv':
        raise ValueError('Please provide the full path of the ZTF .csv file')

    if gd_ztf_h5_path.suffix != '.h5':
        raise ValueError('Please provide the full path of the input .h5 file')
    
    if global_path.suffix != '.csv':
        raise ValueError('Please provide the full path of the globalhost .csv file')
    
    if local_path.suffix != '.csv':
        raise ValueError('Please provide the full path of the localhost .csv file')
    
    if fig_path.suffix != '':
        raise ValueError('Please provide the output folder, not output file path')

    os.makedirs(fig_path, exist_ok = True)

    gd_ztf = load_gibbs_data(gd_ztf_h5_path)

    ztf = pd.read_csv(ztf_csv_path, index_col=[0])
    ztf_hq_vl = ztf.loc[
        (ztf['fitquality_flag'] == 1) & (ztf['lccoverage_flag'] == 1) & # high quality SNe
        (ztf['redshift'] <= 0.06) & (ztf['redshift'] >= 0.015) & # volume limited & no missing SALT fit
        (ztf['sub_type'].isin([ # nonpeculiar SNe
            'norm', 'norm/99aa', '99aa',
            'norm/04gs', '91t', '99aa/91t'
        ]))
    ]
    ztf_hq_vl['mB'] = 10.635-2.5*np.log10(ztf_hq_vl['x0'])

    globalhost = pd.read_csv(global_path)
    globalhost = globalhost.loc[globalhost['ztfname'].isin(ztf_hq_vl['ztfname'])]

    print('globalhost data')
    nan_mask = globalhost[globalhost.columns.to_list()[3:]].isna().any(axis = 1)
    print('rows containing NaNs:', sum(nan_mask))
    print('SNe with nans:')
    print(globalhost.loc[nan_mask, 'ztfname'])

    globalhost.fillna(globalhost.median(numeric_only = True, skipna = True), inplace = True)

    localhost = pd.read_csv(local_path)
    localhost = localhost.loc[localhost['ztfname'].isin(ztf_hq_vl.ztfname)]

    print('localhost data')
    nan_mask = localhost[localhost.columns.to_list()[3:]].isna().any(axis = 1)
    print('rows containing NaNs:', sum(nan_mask))
    print('SNe with nans:')
    print(localhost.loc[nan_mask, 'ztfname'])

    localhost.fillna(localhost.median(numeric_only = True, skipna = True), inplace = True)

    print('Plotting x = globalhost...')
    plot_latent_bias(gd_ztf, host_vec = globalhost.mass, host_vec_err = globalhost.mass_err,
                     xlabel = '$\\log (M / M_☉)_{\\rm glob}$', xval = 10,
                     start_idx = 1000,
                     extra_hlines = {2: (0.03, 'left')},
                     mass_step_labels_loc = 'left',
                     pop1_color = '#3778bf', pop2_color = '#e05c3a',
                     hline_color = '#444444', vline_color = '#444444').savefig(
                        fig_path / Path('ztf_latents_globmass.pdf')
                     )

    print('Plotting x = globalhost, c = ddlr...')
    plot_latent_bias(gd_ztf, host_vec = globalhost.mass, host_vec_err = globalhost.mass_err,
                     xlabel = '$\\log (M / M_☉)_{\\rm glob}$', xval = 10,
                     color_vec = globalhost.d_dlr, color_vec_split_value = 1,
                     start_idx = 1000, clabel = '$dDLR$',
                     extra_hlines = {2: (0.03, 'left')},
                     mass_step_labels_loc = 'left',
                     pop1_color = '#3778bf', pop2_color = '#e05c3a',
                     hline_color = '#444444', vline_color = '#444444').savefig(
                        fig_path / Path('ztf_latents_globmass_ddlr.pdf')
                     )

    print('Plotting x = ddlr...')
    plot_latent_bias(gd_ztf, host_vec = globalhost.d_dlr,
                     xlabel = '$dDLR$', xval = 1,
                     start_idx = 1000,
                     x_min = -0.1,
                     extra_hlines = {2: (0.03, 'right')},
                     mass_step_labels_loc = 'right',
                     pop1_color = '#2a9d8f', pop2_color = '#e76f51',
                     hline_color = '#555555', vline_color = '#555555').savefig(
                        fig_path / Path('ztf_latents_ddlr.pdf')
                     )

    print('Plotting x = ddlr, c = globalhost...')
    plot_latent_bias(gd_ztf, host_vec = globalhost.d_dlr,
                     xlabel = '$dDLR$', xval = 1,
                     color_vec = globalhost.mass, color_vec_split_value = 10,
                     start_idx = 1000, clabel = '$\\log (M / M_☉)_{\\rm glob}$',
                     x_min = -0.1,
                     extra_hlines = {2: (0.03, 'right')},
                     mass_step_labels_loc = 'right',
                     pop1_color = '#2a9d8f', pop2_color = '#e76f51',
                     hline_color = '#555555', vline_color = '#555555').savefig(
                        fig_path / Path('ztf_latents_ddlr_globmass.pdf')
                     )


if __name__ == '__main__':
    ztf_gibbs_latent_plot()
