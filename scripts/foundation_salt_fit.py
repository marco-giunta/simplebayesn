import numpy as np
import pandas as pd
import sncosmo
import astropy
from tqdm import tqdm
import os
from pathlib import Path
from argparse import ArgumentParser

# data obtained using git clone https://github.com/djones1040/Foundation_DR1.git
# The official cosmo sample SALT2 fit can be read using:
# df = astropy.table.Table.read('./Foundation_DR1/Foundation_DR1.FITRES.TEXT', format='ascii').to_pandas()
# Here we replicate the SALT fit for both the cosmo and no cosmo samples.

def foundation_salt_fit(argv = None):
    parser = ArgumentParser(
        description = 'Compute SALT2 fit of Foundation DR1 data (both cosmo and no cosmo samples)'
    )
    parser.add_argument('-i', '--input', type = str, help = 'Path of the Foundation_DR1 folder')
    parser.add_argument('-o', '--output', type = str, help = 'Path of output .csv file')
    parser.add_argument('-s', '--salt', type = str, help = 'Path of SALT2.JLA-B14 snana folder (source: snana/models/SALT2/SALT2.JLA-B14)')

    args = parser.parse_args(argv)

    fnd_folder_path = Path(args.input)
    fnd_csv_path = Path(args.output)
    salt2_jla_b14_path = Path(args.salt)

    if fnd_folder_path.suffix != '':
        raise ValueError('Please provide the input folder, not a file path')
    
    if fnd_csv_path.suffix != '.csv':
        raise ValueError('Please provide the output .csv file path')
    
    os.makedirs(fnd_csv_path.parent, exist_ok = True)

    filter_trans = astropy.io.fits.open(fnd_folder_path / 'kcor_PS1_none.fits')[5].data
    # As stated in the Foundation_DR1 github readme,
    # Foundation uses the standard PS1 bandpass filters for riz, but not g.
    # This can be checked as follows:
    # for b in ['g', 'r', 'i', 'z']:
    #     v = np.allclose(sncosmo.get_bandpass(f'ps1::{b}')(filter_trans['wavelength (A)']), filter_trans[f'ps1-{b}'])
    #     print(b, v)

    # corrected PS1 g band
    sncosmo.register(sncosmo.Bandpass(filter_trans['wavelength (A)'], filter_trans[f'ps1-g'], name = 'ps1::g_cor'))

    def read_fnd_fit(path, use_kcor_ps1_none: bool = True, zpsys = 'ab'): # ps1 photometry uses ab system
        d, t = sncosmo.read_snana_ascii(str(path), 'OBS') # Path object is not supported
        t = t['OBS']
        t.rename_columns(['FLUXCAL', 'FLUXCALERR'], ['flux', 'fluxerr'])
        d['REDSHIFT_HELIO_ERR'] = float(open(path).readlines()[7].split()[3])

        if use_kcor_ps1_none:
            idx = (t['FLT'] == 'g')
            t['FLT'][idx] += '_cor'

        t['FLT'] = 'ps1::' + t['FLT']
        t['zp'] = 27.5 # specified in the foundation readme
        t['zpsys'] = zpsys
        return d, t

    def parse_fit_result(res): # ['z', 't0', 'x0', 'x1', 'c', 'mwebv']
        return {
            'redshift':res.parameters[0],
            'redshift_err':res.zerr,
            't0':res.parameters[1],
            'x0':res.parameters[2],
            'x1':res.parameters[3],
            'c':res.parameters[4],
            'cov_t0_x0':res.covariance[0, 1],
            'cov_t0_x1':res.covariance[0, 2],
            'cov_t0_c':res.covariance[0, 3],
            'cov_x0_x1':res.covariance[1, 2],
            'cov_x0_c':res.covariance[1, 3],
            'cov_x1_c':res.covariance[2, 3],
            **{f'{p}_err':res.errors[p] for p in ['t0', 'x0', 'x1', 'c']}
        }

    # salt2 model and mw dust color law specified in snfit_Foundation_DR1.nml
    def fit_fnd_sn(path, z_type = 'HELIO', dust = sncosmo.F99Dust(), modelcov = True):
        data, table = read_fnd_fit(path)
        model = sncosmo.Model(source = sncosmo.SALT2Source(salt2_jla_b14_path),
                              effects = [dust],
                              effect_names = ['mw'],
                              effect_frames = ['obs'])
        model.set(z = data[f'REDSHIFT_{z_type}'])
        model.set(mwebv = data['MWEBV'])

        result, fitted_model = sncosmo.fit_lc(
            table, model, ['t0','x0','x1','c'], modelcov = modelcov,
        )
        result.zerr = data['REDSHIFT_HELIO_ERR']
        return parse_fit_result(result)

    def fit_fnd_sample(sample_path: Path):
        sn_data_list = []
        file_names = os.listdir(sample_path)
        # ignore Foundation_DR1.IGNORE, Foundation_DR1.LIST, Foundation_DR1.README;
        # all valid SNe are called Foundation_DR1_(...)

        for f in tqdm(file_names):
            if f.startswith('Foundation_DR1.'):
                continue
            sn_data_list.append({'id':f.removeprefix('Foundation_DR1_').removesuffix('.txt'),
                                 **fit_fnd_sn(sample_path / f)})

        return pd.DataFrame(sn_data_list)

    print('Fitting cosmo sample...')
    fnd_cosmo = fit_fnd_sample(fnd_folder_path / 'Foundation_DR1')
    fnd_cosmo['sample'] = 'cosmo'

    print('Fitting no cosmo sample...')
    fnd_no_cosmo = fit_fnd_sample(fnd_folder_path / 'Foundation_DR1_NoCosmo')
    fnd_no_cosmo['sample'] = 'no_cosmo'

    fnd_all = pd.concat([fnd_cosmo, fnd_no_cosmo], ignore_index = True)
    fnd_all['mB'] = 10.635-2.5*np.log10(fnd_all['x0'])
    fnd_all.to_csv(fnd_csv_path, index = False)

if __name__ == '__main__':
    foundation_salt_fit()
