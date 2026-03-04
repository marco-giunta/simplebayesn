import pandas as pd
import simplebayesn
import os
from pathlib import Path
from argparse import ArgumentParser

def foundation_gibbs(argv = None):
    parser = ArgumentParser(
        description = 'Fit SimpleBayeSN Gibbs sampler on Foundation DR1 data (cosmo + no cosmo)'
    )

    parser.add_argument('-i', '--input', type = str, help = 'Path of Foundation DR1 cosmo + no cosmo .csv (input file)')
    parser.add_argument('-o', '--output', type = str, help = 'Path of GibbsData .h5 MCMC output file')

    args = parser.parse_args(argv)

    fnd_csv_path = Path(args.input)
    gd_fnd_h5_path = Path(args.output)

    if fnd_csv_path.suffix != '.csv':
        raise ValueError('Please provide an input .csv file')

    if gd_fnd_h5_path.suffix != '.h5':
        raise ValueError('Please provide the full path of the output .h5 file')

    os.makedirs(gd_fnd_h5_path.parent, exist_ok = True)

    fnd = pd.read_csv(fnd_csv_path, index_col = False)

    # Like in Foley et al. 2018, we discard the 12 peculiar SNe,
    # but redo the salt fit and apply the ZTF color/stretch cuts instead.
    # We do not apply the other cuts (e.g. model-dependent Chauvenet criterion).

    # Notice that each Foundation SN has multiple ids,
    # and the one chosen to name the file in the cosmo/no cosmo folders is not necessarily the same used in a certain part of the paper,
    # so the ids must be matched separately.

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

    fnd_cuts = fnd.loc[
        (~fnd['id'].isin(pec_ids)) &
        (fnd['c'] >= -0.2) & (fnd['c'] <= 0.8) &
        (fnd['x1'] >= -3) & (fnd['x1'] <= 3)
    ]

    sd_fnd = simplebayesn.preprocess_data(fnd_cuts)
    print(f'Discarded Foundation SNe: {len(fnd)-sd_fnd.num_samples}/{len(fnd)} (of which {len(pec_ids)} discarded due to SN peculiar type)')

    prior_params = simplebayesn.priors.gibbs.get_priors_params_uniform_invgamma()
    iv = simplebayesn.initialize.sample_initial_values_uniform(num_samples = sd_fnd.num_samples, seed = 1234)
    gd_fnd = simplebayesn.samplers.gibbs_sampler(iv, prior_params, sd_fnd, num_iter = int(1e5), seed = 1234)

    gd_fnd.save(gd_fnd_h5_path)

    return gd_fnd

if __name__ == '__main__':
    foundation_gibbs()
