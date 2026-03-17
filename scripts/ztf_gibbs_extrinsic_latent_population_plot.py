import simplebayesn
import os
from pathlib import Path
from argparse import ArgumentParser

def ztf_gibbs_extrinsic_population_latent_plot(argv = None):
    parser = ArgumentParser(
        description = 'Plot latent extinguished stretch corrected absolute magnitude vs latent extinguished color using Gibbs MCMC chain of ZTF HQ VL data'
    )
    parser.add_argument('-i', '--input', type = str, help = 'Path of GibbsData .h5 (input file)')
    parser.add_argument('-o', '--output', type = str, help = 'Path of the folder where to save the output plot')

    args = parser.parse_args(argv)

    gd_ztf_h5_path = Path(args.input)
    fig_path = Path(args.output)

    if gd_ztf_h5_path.suffix != '.h5':
        raise ValueError('Please provide the full path of the input .h5 file')
    
    if fig_path.suffix != '':
        raise ValueError('Please provide the output folder, not output file path')

    os.makedirs(fig_path, exist_ok = True)

    gd_ztf = simplebayesn.load_gibbs_data(gd_ztf_h5_path)

    print('Plotting extrinsic latent population...')
    fig, ax = simplebayesn.visualize.extinguished_magnitude_color_distribution_mean(gd_ztf, 1000, color_dust=True)
    fig.savefig(fig_path / Path('ztf_extrinsic_latent_population_plot.pdf'))

if __name__ == '__main__':
    ztf_gibbs_extrinsic_population_latent_plot()

# To show an interactive version of this plot instead of a single static frame, run the following in a jupyter notebook
# (focus on a subset of the MCMC samples with a large stride to keep the animation size reasonable)

# import matplotlib.pyplot as plt
# from IPython.display import HTML
# anim, fig = simplebayesn.visualize.extinguished_magnitude_color_distribution_animation(gd_ztf, start_idx=20000, stop_idx=40000, step_stride=500, color_dust=True)
# plt.close(fig)
# HTML(anim.to_jshtml())
