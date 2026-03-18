import simplebayesn
import os
from pathlib import Path
from argparse import ArgumentParser

LATEX_MAP = {
    'd_dlr':'dDLR$',
    'mass':'\\log (M_* / M_☉)', # \astrosun
    'restframe_gz':'(g-z)'
}

def plot_ztf_gibbs_subset_cornerplot(ztf_less, ztf_larger, ztf_complete, name, value, kind):
    label = f'${LATEX_MAP[name]}' + '_{\\rm ' + f'{kind}' + '}$'
    fig = simplebayesn.visualize.compare_posterior_cornerplots([ztf_less, ztf_larger, ztf_complete], start_idx = 1000, labels = [f'{label} < {value:.1f}', f'{label} > {value:.1f}', 'all'])
    return fig

def plot_ztf_gibbs_subset_cornerplot_color(ztf_less, ztf_larger, ztf_complete, name, value, kind):
    pp = ['tau', 'RB', 'c0_int', 'sigmac_int2', 'M0_int', 'beta_int']
    label = f'${LATEX_MAP[name]}' + '_{\\rm ' + f'{kind}' + '}$'
    fig = simplebayesn.visualize.compare_posterior_cornerplots([ztf_less, ztf_larger, ztf_complete], start_idx = 1000, labels = [f'{label} < {value:.1f}', f'{label} > {value:.1f}', 'all'],
    params_to_plot = pp)
    return fig

def ztf_gibbs_mass_splits_posterior_cornerplots(argv = None):
    parser = ArgumentParser(
        description = 'Plot posterior cornerplots of Gibbs MCMC chains of ZTF HQ VL data subsets, partitioned according to available environmental proxies'
    )
    parser.add_argument('-i', '--input', type = str, help = 'Path of the folder containing the .h5 chains from ZTF data subsets (input files)')
    parser.add_argument('-iz', '--input_ztf', type = str, help = 'Path of the .h5 file corresponding to the Gibbs run on the complete ZTF HQ VL data')
    parser.add_argument('-o', '--output', type = str, help = 'Path of the folder where to save the output cornerplots')

    args = parser.parse_args(argv)

    ztf_complete_h5_path = Path(args.input_ztf)
    ztf_subset_path = Path(args.input)
    fig_path = Path(args.output)

    if ztf_complete_h5_path.suffix != '.h5':
        raise ValueError('Please provide the input ZTF .h5 file for the complete HQ VL dataset')

    if ztf_subset_path.suffix != '':
        raise ValueError('Please provide the path of the folder containing the set of .h5 files for the ZTF HQ VL subsets')

    if fig_path.suffix != '':
        raise ValueError('Please provide the path of the folder where to save the output plots')

    os.makedirs(fig_path, exist_ok = True)

    gd_ztf = simplebayesn.load_gibbs_data(ztf_complete_h5_path)

    global_vars = {
        'mass': 10,
        'd_dlr': 1.5,
        'restframe_gz': 1,
    }

    local_vars = {
        'mass': 8.9,
        'restframe_gz': 1
    }

    for k, v in global_vars.items():
        print(f'Plotting global {k}...')
        plot_ztf_gibbs_subset_cornerplot(
            simplebayesn.load_gibbs_data(
                ztf_subset_path / Path(f'ztf_global_{k}_less_than_{v}.h5')
            ),
            simplebayesn.load_gibbs_data(
                ztf_subset_path / Path(f'ztf_global_{k}_larger_than_{v}.h5')
            ),
            gd_ztf,
            k, v, 'glob'
        ).savefig(
            fig_path / Path(f'ztf_global_{k}.pdf')
        )
        print(f'Plotting global {k} (color parameters)...')
        plot_ztf_gibbs_subset_cornerplot_color(
            simplebayesn.load_gibbs_data(
                ztf_subset_path / Path(f'ztf_global_{k}_less_than_{v}.h5')
            ),
            simplebayesn.load_gibbs_data(
                ztf_subset_path / Path(f'ztf_global_{k}_larger_than_{v}.h5')
            ),
            gd_ztf,
            k, v, 'glob'
        ).savefig(
            fig_path / Path(f'ztf_global_{k}_color.pdf')
        )

    for k, v in local_vars.items():
        print(f'Plotting local {k}...')
        plot_ztf_gibbs_subset_cornerplot(
            simplebayesn.load_gibbs_data(
                ztf_subset_path / Path(f'ztf_local_{k}_less_than_{v}.h5')
            ),
            simplebayesn.load_gibbs_data(
                ztf_subset_path / Path(f'ztf_local_{k}_larger_than_{v}.h5')
            ),
            gd_ztf,
            k, v, 'loc'
        ).savefig(
            fig_path / Path(f'ztf_local_{k}.pdf')
        )
        print(f'Plotting local {k} (color parameters)...')
        plot_ztf_gibbs_subset_cornerplot_color(
            simplebayesn.load_gibbs_data(
                ztf_subset_path / Path(f'ztf_local_{k}_less_than_{v}.h5')
            ),
            simplebayesn.load_gibbs_data(
                ztf_subset_path / Path(f'ztf_local_{k}_larger_than_{v}.h5')
            ),
            gd_ztf,
            k, v, 'loc'
        ).savefig(
            fig_path / Path(f'ztf_local_{k}_color.pdf')
        )

if __name__ == '__main__':
    ztf_gibbs_mass_splits_posterior_cornerplots()
