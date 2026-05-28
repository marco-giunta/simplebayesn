import simplebayesn
from pathlib import Path
import os
from argparse import ArgumentParser

def ztf_sim_sel_cornerplot(argv = None):
    parser = ArgumentParser(
        description = 'Compare posterior cornerplots of renormalized emcee and non-renormalized Simple-BayeSN Gibbs sampler on ZTF HQ VL-like simulated data'
    )
    parser.add_argument('-ie', '--input_emcee', type = str, help = 'Path of the emcee .h5 data (input file)')
    parser.add_argument('-i3', '--input_gibbs3', type = str, help = 'Path of the Gibbs .h5 data for the c<=0.3 cut (input file)')
    parser.add_argument('-i8', '--input_gibbs8', type = str, help = 'Path of the Gibbs .h5 data for the c<=0.8 cut (input file)')
    parser.add_argument('-o', '--output', type = str, help = 'Path of the folder where to save the cornerplot')

    args = parser.parse_args(argv)

    ge_ztf_h5_path = Path(args.input_emcee)
    gd08_ztf_h5_path = Path(args.input_gibbs8)
    gd03_ztf_h5_path = Path(args.input_gibbs3)
    fig_path = Path(args.output)

    if ge_ztf_h5_path.suffix != '.h5':
        raise ValueError('Please provide the full path of the emcee .h5 file')

    if gd08_ztf_h5_path.suffix != '.h5':
        raise ValueError('Please provide the full path of the Gibbs 0.8 .h5 file')

    if gd03_ztf_h5_path.suffix != '.h5':
        raise ValueError('Please provide the full path of the Gibbs 0.3 .h5 file')

    if fig_path.suffix != '':
        raise ValueError('Please provide the output folder, not output file path')

    os.makedirs(fig_path, exist_ok = True)

    ge = simplebayesn.load_emcee_data(ge_ztf_h5_path)
    gd08 = simplebayesn.load_gibbs_data(gd08_ztf_h5_path)
    gd03 = simplebayesn.load_gibbs_data(gd03_ztf_h5_path)

    gp_true = {
        'tau': 0.15,
        'RB': 3.26,
        'x0': -0.27,
        'sigmax2': 1.07,
        'c0_int': -0.08,
        'alphac_int': -0.006,
        'sigmac_int2': (0.038)**2,
        'M0_int': -19.48,
        'alpha': -0.16,
        'beta_int': 0,
        'sigma_int2': (0.11)**2
    }

    colors = ['#4c72b0', '#55a868', '#c44e52']
    labels = ['Renormalised, $\\hat{c} \\leq 0.3$',
              'Non-renormalised, $\\hat{c} \\leq 0.3$',
              'Non-renormalised, $\\hat{c} \\leq 0.8$']
    chains = [ge, gd08, gd03]

    fig = simplebayesn.visualize.compare_posterior_cornerplots(chains, 1000, labels=labels,
                                                               truth_dict=gp_true,
                                                               contours_colors=colors);
    fig.savefig(fig_path / Path('ztf_sim_bias.pdf'))

if __name__ == '__main__':
    ztf_sim_sel_cornerplot()
