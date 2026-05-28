import pandas as pd
import simplebayesn
from pathlib import Path
from argparse import ArgumentParser

def ztf_sim_sel_gibbs(argv = None):
    parser = ArgumentParser(
        description = 'Fit non-renormalized Simple-BayeSN Gibbs sampler on ZTF HQ VL-like simulated data'
    )
    parser.add_argument('-i', '--input', type = str, help = 'Path of the ZTF .csv (input file)')
    parser.add_argument('-o8', '--output8', type = str, help = 'Path of the Gibbs data .h5 MCMC output file for the c<=0.8 cut')
    parser.add_argument('-o3', '--output3', type = str, help = 'Path of the Gibbs data .h5 MCMC output file for the c<=0.3 cut')

    args = parser.parse_args(argv)

    ztf_csv_path = Path(args.input)
    gd08_ztf_h5_path = Path(args.output8)
    gd03_ztf_h5_path = Path(args.output3)

    if ztf_csv_path.suffix != '.csv':
        raise ValueError('Please provide an input .csv file')

    if gd08_ztf_h5_path.suffix != '.h5':
        raise ValueError('Please provide the full path of the output8 .h5 file')

    if gd03_ztf_h5_path.suffix != '.h5':
        raise ValueError('Please provide the full path of the output3 .h5 file')

    ztf = pd.read_csv(ztf_csv_path, index_col=[0])
    ztf_hq_vl = ztf.loc[
        (ztf['fitquality_flag'] == 1) & (ztf['lccoverage_flag'] == 1) & # high quality SNe
        (ztf['redshift'] <= 0.06) & (ztf['redshift'] >= 0.015) & # volume limited & no missing SALT fit
        (ztf['sub_type'].isin([ # nonpeculiar SNe
            'norm', 'norm/99aa', '99aa',
            'norm/04gs', '91t', '99aa/91t'
        ]))
    ]
    sd_ztf = simplebayesn.preprocess_data(ztf_hq_vl)

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

    sim08 = simplebayesn.simulators.sbsn.simulate_simplebayesn_salt_data_from_redshift_cov_arr(
        sd_ztf.z,
        sd_ztf.sigma_z,
        sd_ztf.cov,
        gp_true,
        0,
        clim = (-0.3, 0.8),
        xlim = (-3, 3)
    )
    sd_sim08 = sim08['observed_data']

    sim03 = simplebayesn.simulators.sbsn.simulate_simplebayesn_salt_data_from_redshift_cov_arr(
        sd_ztf.z,
        sd_ztf.sigma_z,
        sd_ztf.cov,
        gp_true,
        0,
        clim = (-0.3, 0.3),
        xlim = (-3, 3)
    )
    sd_sim03 = sim03['observed_data']

    iv = simplebayesn.initialize.sample_initial_values_uniform(sd_sim03.num_samples, 1234, allow_negative_beta_int=True)
    priors = simplebayesn.priors.gibbs.get_priors_params_uniform_invgamma()
    print('fitting 0.3 cuts')
    gd03 = simplebayesn.samplers.gibbs_sampler(iv, priors, sd_sim03, int(1e5), 1234)
    gd03.save(gd03_ztf_h5_path)

    iv = simplebayesn.initialize.sample_initial_values_uniform(sd_sim08.num_samples, 1234, allow_negative_beta_int=True)
    print('fitting 0.8 cuts')
    gd08 = simplebayesn.samplers.gibbs_sampler(iv, priors, sd_sim08, int(1e5), 1234)
    gd08.save(gd08_ztf_h5_path)

if __name__ == '__main__':
    ztf_sim_sel_gibbs()
