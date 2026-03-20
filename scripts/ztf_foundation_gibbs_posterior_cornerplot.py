import simplebayesn
import os
from pathlib import Path
from argparse import ArgumentParser

def ztf_foundation_gibbs_posterior_cornerplot(argv = None):
    parser = ArgumentParser(
        description = 'Plot posterior cornerplots of Gibbs MCMC chains of ZTF HQ VL vs Foundation cosmo + no cosmo data'
    )
    parser.add_argument('-iz', '--input_ztf', type = str, help = 'Path of ZTF GibbsData .h5 (input file)')
    parser.add_argument('-ifn', '--input_fnd_nc', type = str, help = 'Path of Foundation cosmo + no cosmo GibbsData .h5 (input file)')
    parser.add_argument('-ifc', '--input_fnd_c', type = str, help = 'Path of Foundation cosmo only GibbsData .h5 (input file)')
    parser.add_argument('-o', '--output', type = str, help = 'Path of the folder where to save the output cornerplots')

    args = parser.parse_args(argv)

    gd_ztf_h5_path = Path(args.input_ztf)
    gd_fnd_nc_h5_path = Path(args.input_fnd_nc)
    gd_fnd_c_h5_path = Path(args.input_fnd_c)
    fig_path = Path(args.output)

    if gd_ztf_h5_path.suffix != '.h5':
        raise ValueError('Please provide the full path of the input .h5 file')

    if gd_fnd_nc_h5_path.suffix != '.h5':
        raise ValueError('Please provide the full path of the input .h5 file')
    
    if gd_fnd_c_h5_path.suffix != '.h5':
        raise ValueError('Please provide the full path of the input .h5 file')

    if fig_path.suffix != '':
        raise ValueError('Please provide the output folder, not output file path')

    os.makedirs(fig_path, exist_ok = True)

    gd_ztf = simplebayesn.load_gibbs_data(gd_ztf_h5_path)
    gd_fnd_nc = simplebayesn.load_gibbs_data(gd_fnd_nc_h5_path)
    gd_fnd_c = simplebayesn.load_gibbs_data(gd_fnd_c_h5_path)

    print('Plotting complete cornerplot...')
    fig = simplebayesn.visualize.compare_posterior_cornerplots([gd_fnd_nc, gd_ztf, gd_fnd_c], start_idx = 1000,
                                                                labels = ['Foundation (cosmo + no cosmo)',
                                                                          'ZTF HQ VL',
                                                                          'Foundation (cosmo)'],
                                                                contours_colors = ['C0', 'C1', 'C2'])
    fig.savefig(fig_path / Path('ztf_fnd_cornerplot.pdf'))

    print('Plotting color cornerplot...')
    pp = ['tau', 'RB', 'c0_int', 'sigmac_int', 'beta_int']
    fig = simplebayesn.visualize.compare_posterior_cornerplots([gd_fnd_nc, gd_ztf, gd_fnd_c], start_idx = 1000, params_to_plot = pp,
                                                                labels = ['Foundation (cosmo + no cosmo)',
                                                                          'ZTF HQ VL',
                                                                          'Foundation (cosmo)'],
                                                                contours_colors = ['C0', 'C1', 'C2'])
    fig.savefig(fig_path / Path('ztf_fnd_color_params_cornerplot.pdf'))

if __name__ == '__main__':
    ztf_foundation_gibbs_posterior_cornerplot()
