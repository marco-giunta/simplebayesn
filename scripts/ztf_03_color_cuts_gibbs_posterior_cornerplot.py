import simplebayesn
import os
from pathlib import Path
from argparse import ArgumentParser

def ztf_gibbs_posterior_cornerplot(argv = None):
    parser = ArgumentParser(
        description = 'Plot posterior cornerplots of Gibbs MCMC chain of ZTF HQ VL data'
    )
    parser.add_argument('-i', '--input', type = str, help = 'Path of complete ZTF GibbsData .h5 (input file)')
    parser.add_argument('-ic', '--input_cuts', type = str, help = 'Path of c<=0.3 cut ZTF GibbsData .h5 (input file)')
    parser.add_argument('-o', '--output', type = str, help = 'Path of the folder where to save the output cornerplots')

    args = parser.parse_args(argv)

    gd_ztf_h5_path = Path(args.input)
    gd_ztf_03_h5_path = Path(args.input_cuts)
    fig_path = Path(args.output)

    if gd_ztf_h5_path.suffix != '.h5':
        raise ValueError('Please provide the full path of the input .h5 file')

    if gd_ztf_03_h5_path.suffix != '.h5':
        raise ValueError('Please provide the full path of the input .h5 file')

    if fig_path.suffix != '':
        raise ValueError('Please provide the output folder, not output file path')

    os.makedirs(fig_path, exist_ok = True)

    gd_ztf = simplebayesn.load_gibbs_data(gd_ztf_h5_path)
    gd_ztf_03 = simplebayesn.load_gibbs_data(gd_ztf_03_h5_path)

    print('Plotting complete cornerplot...')
    fig = simplebayesn.visualize.compare_posterior_cornerplots(
        [gd_ztf, gd_ztf_03], start_idx = 1000,
        contours_colors = ['C1', 'lightskyblue'],
        labels = ['ZTF HQ VL, $\\hat c \\leq 0.8$', 'ZTF HQ VL, $\\hat c \\leq 0.3$']
    )
    fig.savefig(fig_path / Path('ztf_03_cornerplot.pdf'))

    print('Plotting color cornerplot...')
    pp = ['tau', 'RB', 'c0_int', 'sigmac_int', 'beta_int']
    fig = simplebayesn.visualize.compare_posterior_cornerplots(
        [gd_ztf, gd_ztf_03], start_idx = 1000,
        contours_colors = ['C1', 'lightskyblue'],
        labels = ['ZTF HQ VL, $\\hat c \\leq 0.8$', 'ZTF HQ VL, $\\hat c \\leq 0.3$'],
        params_to_plot = pp
    )
    fig.savefig(fig_path / Path('ztf_color_params_03_cornerplot.pdf'))

if __name__ == '__main__':
    ztf_gibbs_posterior_cornerplot()
