import numpy as np
import pandas as pd
import simplebayesn
from pathlib import Path
from argparse import ArgumentParser

def ztf_sim_sel_emcee(argv = None):
    parser = ArgumentParser(
        description = 'Fit renormalized Simple-BayeSN emcee sampler on ZTF HQ VL-like simulated data'
    )
    parser.add_argument('-i', '--input', type = str, help = 'Path of the ZTF .csv (input file)')
    parser.add_argument('-o', '--output', type = str, help = 'Path of the emcee data .h5 MCMC output file')

    args = parser.parse_args(argv)

    ztf_csv_path = Path(args.input)
    ge_ztf_h5_path = Path(args.output)

    if ztf_csv_path.suffix != '.csv':
        raise ValueError('Please provide an input .csv file')

    if ge_ztf_h5_path.suffix != '.h5':
        raise ValueError('Please provide the full path of the output .h5 file')


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

    sim = simplebayesn.simulators.sbsn.simulate_simplebayesn_salt_data_from_redshift_cov_arr(
        sd_ztf.z,
        sd_ztf.sigma_z,
        sd_ztf.cov,
        gp_true,
        0,
        clim = (-0.3, 0.3),
        xlim = (-3, 3)
    )
    sd_sim = sim['observed_data']

    nw = 25
    iv = simplebayesn.initialize.sample_initial_values_uniform(
        sd_sim.num_samples, seed=np.arange(nw),
        marginal=True, to_param_array=True,
        allow_negative_beta_int=True,
    )

    lp = simplebayesn.priors.emcee.uniform_invgamma_marginal_log_prior

    simplebayesn.samplers.emcee_sampler(
        nw, 1000, int(1e4),
        iv, lp, sd_sim,
        selection='mc',
        clim=(-0.3, 0.3), xlim=(-3, 3),
        num_sim_per_sample=2000,
        path=ge_ztf_h5_path
    )

if __name__ == '__main__':
    ztf_sim_sel_emcee()
