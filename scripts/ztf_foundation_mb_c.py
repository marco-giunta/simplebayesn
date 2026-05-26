import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde
from astropy.table import Table
import os
from argparse import ArgumentParser
from pathlib import Path

PERCENT_SIZE = 25
FILL_ALPHA = 0.15
ZTF_COLOR = 'C1'
FND_COSMO_COLOR = 'C0'
FND_NOCOSMO_COLOR = 'C2'
C_MIN_PLOT, C_MAX_PLOT = -0.3, 0.9

def ztf_foundation_mb_c(argv = None):
    parser = ArgumentParser(
        description = 'Scatter plot ZTF HQ VL vs Foundation DR1 (cosmo + no cosmo) in magnitude/color space'
    )
    parser.add_argument('-iz', '--input_ztf', type = str, help = 'Path of ZTF .csv (input file)')
    parser.add_argument('-iff', '--input_fnd_fitres', type = str, help = 'Path of Foundation .FITRES.TEXT file (official, cosmo only)')
    parser.add_argument('-ifc', '--input_fnd_csv', type = str, help = 'Path of Foundation .csv file (salt fit from this scripts folder, cosmo + no cosmo)')
    parser.add_argument('-o', '--output', type = str, help = 'Path of the folder where to save the output scatterplot')

    args = parser.parse_args(argv)

    ztf_csv_path = Path(args.input_ztf)
    fnd_fitres_path = Path(args.input_fnd_fitres)
    fnd_csv_path = Path(args.input_fnd_csv)
    fig_path = Path(args.output)

    if ztf_csv_path.suffix != '.csv':
        raise ValueError('Please provide the input ZTF .csv file')

    if fnd_fitres_path.suffix != '.TEXT':
        raise ValueError('Please provide the input Foundation .FITRES.TEXT file')

    if fnd_csv_path.suffix != '.csv':
        raise ValueError('Please provide the input Foundation SALT2 fit .csv data')

    if fig_path.suffix != '':
        raise ValueError('Please provide the path folder where to save the output, not the complete path including the file name')

    os.makedirs(fig_path, exist_ok = True)

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

    fnd_cosmo = Table.read(fnd_fitres_path, format='ascii').to_pandas()
    fnd_cosmo = fnd_cosmo.loc[
        (fnd_cosmo['c'] >= -0.2) & (fnd_cosmo['c'] <= 0.8) &
        (fnd_cosmo['x1'] >= -3) & (fnd_cosmo['x1'] <= 3)
    ]

    pec_ids = [
        'ASASSN-15ga',
        'ASASSN-15hy',
        'ASASSN-16ci',
        'ASASSN-16ex',
        'MASTERJ2222',
        'PS16amf',
        'PSNJ1300323',
        'PSNJ1628383',
        'PSNJ2310226',
        'SN2016ajf',
        'SN2017lc',
        'SN2017mu'
    ]

    fnd_nocosmo = pd.read_csv(fnd_csv_path)
    fnd_nocosmo = fnd_nocosmo.loc[
        (fnd_nocosmo['sample'] == 'no_cosmo') & (~fnd_nocosmo['id'].isin(pec_ids)) &
        (fnd_nocosmo['c'] >= -0.2) & (fnd_nocosmo['c'] <= 0.8) &
        (fnd_nocosmo['x1'] >= -3) & (fnd_nocosmo['x1'] <= 3)
    ]
    fnd_nocosmo['mB'] = 10.635-2.5*np.log10(fnd_nocosmo['x0'])

    fig, ax = plt.subplots(1, 1, figsize = (12, 7))

    ax.scatter(ztf_hq_vl.c, ztf_hq_vl.mB, alpha = 0.3, s = 10, label = f'ZTF HQ VL ({len(ztf_hq_vl)} SNe)', color = ZTF_COLOR)
    ax.scatter(fnd_cosmo.c, fnd_cosmo.mB, alpha = 0.5, s = 20, label = f'Foundation (cosmo) ({len(fnd_cosmo)} SNe)', color = FND_COSMO_COLOR, edgecolors = 'k', linewidths = 0.5)
    ax.scatter(fnd_nocosmo.c, fnd_nocosmo.mB, alpha = 0.8, s = 30, label = f'Foundation (no cosmo) ({len(fnd_nocosmo)} SNe)', color = FND_NOCOSMO_COLOR, marker = 'x', linewidths = 1.5)

    ax.set_xlim(C_MIN_PLOT, C_MAX_PLOT)
    ax.axvline(0.3, color = 'gray', linestyle = '--', alpha = 0.5, label = '$\\hat c=0.3$')
    ax.axvspan(0.3, C_MAX_PLOT, color = 'gray', alpha = 0.1, zorder = 0)

    ax.set_xlabel('$\\hat{c}$', fontsize = 15)
    ax.set_ylabel('$\\hat{m}_B$', fontsize = 15)
    ax.legend(fontsize = 12)
    ax.invert_yaxis()
    ax.tick_params(axis = 'both', labelsize = 11)

    divider = make_axes_locatable(ax)
    ax_top = divider.append_axes('top', size = f'{PERCENT_SIZE}%', pad = 0.1, sharex = ax)
    ax_right = divider.append_axes('right', size = f'{PERCENT_SIZE}%', pad = 0.1, sharey = ax)

    c_range = np.linspace(C_MIN_PLOT, C_MAX_PLOT, 200) #np.linspace(-0.3, 0.8, 200)
    kde_ztf_c = gaussian_kde(ztf_hq_vl.c)
    kde_fnd_cosmo_c = gaussian_kde(fnd_cosmo.c)
    kde_fnd_nocosmo_c = gaussian_kde(fnd_nocosmo.c)

    # top KDE: color
    ax_top.plot(c_range, kde_ztf_c(c_range), color = ZTF_COLOR, label = 'ZTF HQ VL', linewidth = 2)
    ax_top.plot(c_range, kde_fnd_cosmo_c(c_range), color = FND_COSMO_COLOR, label = 'Foundation (cosmo)', linewidth = 2)
    ax_top.plot(c_range, kde_fnd_nocosmo_c(c_range), color = FND_NOCOSMO_COLOR, label = 'Foundation (no cosmo)', linewidth = 2)
    ax_top.fill_between(c_range, kde_ztf_c(c_range), alpha = FILL_ALPHA, color = ZTF_COLOR)
    ax_top.fill_between(c_range, kde_fnd_cosmo_c(c_range), alpha = FILL_ALPHA, color = FND_COSMO_COLOR)
    ax_top.fill_between(c_range, kde_fnd_nocosmo_c(c_range), alpha = FILL_ALPHA, color = FND_NOCOSMO_COLOR)
    ax_top.axvline(0.3, color = 'gray', linestyle = '--', alpha = 0.5)
    ax_top.axvspan(0.3, C_MAX_PLOT, color = 'gray', alpha = 0.1, zorder = 0)
    ax_top.tick_params(labelbottom = False)
    ax_top.legend(fontsize = 9, loc = 'upper right')

    # right KDE: magnitude (swap x and y because of the rotated axes, or use rotated functions)
    m_range = np.linspace(13, 21, 200)
    kde_ztf_m = gaussian_kde(ztf_hq_vl.mB)
    kde_fnd_cosmo_m = gaussian_kde(fnd_cosmo.mB)
    kde_fnd_nocosmo_m = gaussian_kde(fnd_nocosmo.mB)

    ax_right.plot(kde_ztf_m(m_range), m_range, color = ZTF_COLOR, linewidth = 2)
    ax_right.plot(kde_fnd_cosmo_m(m_range), m_range, color = FND_COSMO_COLOR, linewidth = 2)
    ax_right.plot(kde_fnd_nocosmo_m(m_range), m_range, color = FND_NOCOSMO_COLOR, linewidth = 2)
    ax_right.fill_betweenx(m_range, kde_ztf_m(m_range), alpha = FILL_ALPHA, color = ZTF_COLOR)
    ax_right.fill_betweenx(m_range, kde_fnd_cosmo_m(m_range), alpha = FILL_ALPHA, color = FND_COSMO_COLOR)
    ax_right.fill_betweenx(m_range, kde_fnd_nocosmo_m(m_range), alpha = FILL_ALPHA, color = FND_NOCOSMO_COLOR)
    ax_right.tick_params(labelleft = False)
    ax_right.invert_yaxis()

    fig.savefig(fig_path / Path('ztf_foundation_mb_c.pdf'))

if __name__ == '__main__':
    ztf_foundation_mb_c()
