# simplebayesn

A Python implementation of the **Simple-BayeSN** Bayesian hierarchical model for type Ia supernova population inference, as introduced in [Mandel et al. (2017)](https://doi.org/10.3847/1538-4357/aa6038) and extended in Giunta, Karchev & Trotta (2026).

The model decomposes the observed SN Ia colour-magnitude correlation into an intrinsic Gaussian component and an extrinsic exponential dust component, inferring population-level parameters from SALT2 light-curve summaries via Gibbs and emcee samplers.

---

## Installation

**From GitHub (recommended):**
```bash
pip install git+https://github.com/marco-giunta/simplebayesn.git
```

**For development:**
```bash
git clone https://github.com/marco-giunta/simplebayesn.git
cd simplebayesn
pip install -e .
```

---

## Overview

Simple-BayeSN models each SN Ia as follows. The intrinsic absolute magnitude depends on stretch and intrinsic colour via a linear relation with slopes $\alpha$ and $\beta_{\rm int}$. The intrinsic colour is drawn from a Gaussian with mean $c_0^{\rm int} + \alpha_c^{\rm int} x$ and standard deviation $\sigma_c^{\rm int}$. Host-galaxy dust reddening $E \sim \rm{Exp}(\tau)$ then reddens and dims each SN according to extinction coefficient $R_B$. The resulting observed colour-magnitude distribution is a Gaussian-exponential convolution whose curvature encodes the relative importance of intrinsic and extrinsic effects.

The 11 population-level parameters inferred by the model are:

| Parameter | Description |
|---|---|
| $\tau$ (`tau`) | Mean host-galaxy dust reddening $E(B−V)$ |
| $R_B$ (`RB`) | Ratio of B-band extinction to $E(B−V)$ |
| $M_0^{\rm int}$ (`M0_int`) | Intrinsic SN Ia absolute magnitude offset |
| $\alpha$ (`alpha`) | Stretch-magnitude correlation |
| $\beta_{\rm int}$ (`beta_int`) | Intrinsic colour-magnitude correlation |
| $\sigma_{\rm int}^2$ (`sigma_int2`) | Intrinsic magnitude variance |
| $c_0^{\rm int}$ (`c0_int`) | Mean intrinsic colour at $x=0$ |
| $\alpha_c^{\rm int}$ (`alphac_int`) | Stretch-colour correlation |
| $(\sigma_c^{\rm int})^2$ (`sigmac_int2`) | Intrinsic colour variance |
| $x_0$ (`x0`) | Mean stretch |
| $\sigma_x^2$ (`sigmax2`) | Stretch variance |

---

## Usage

### Preprocessing

```python
import simplebayesn
import pandas as pd

# Load a DataFrame with SALT2 fit results.
# Required columns: x0, c, x1, redshift, redshift_err,
#                   x0_err, c_err, x1_err, cov_x0_c, cov_x0_x1, cov_x1_c

df = pd.read_csv('./ztf/ztfsniadr2/tables/snia_data.csv') # API modelled after the ZTF dataframe conventions
salt_data = simplebayesn.preprocess_data(df) # this caches distmod astropy calls and computes their linearized errors
```

### Gibbs sampling

```python
prior_params = simplebayesn.priors.gibbs.get_priors_params_uniform_invgamma()
iv = simplebayesn.initialize.sample_initial_values_uniform(
    num_samples=salt_data.num_samples,
    seed=42,
    allow_negative_beta_int=True
)
gibbs_data = simplebayesn.samplers.gibbs_sampler(iv, prior_params, salt_data, num_iter=100_000, seed=42)
gibbs_data.save('output_gibbs.h5')
```

### emcee sampling (marginal likelihood)

```python
import numpy as np

nwalkers = 25
iv_emcee = simplebayesn.initialize.sample_initial_values_uniform(
    salt_data.num_samples, seed=np.arange(nwalkers),
    marginal=True, to_param_array=True,
    allow_negative_beta_int=True
)
log_prior = simplebayesn.priors.emcee.uniform_invgamma_marginal_log_prior
simplebayesn.samplers.emcee_sampler(
    nwalkers, 1000, 10_000,
    iv_emcee, log_prior, salt_data,
    path='output_emcee.h5'
)
```

### Loading results

```python
gibbs_data = simplebayesn.load_gibbs_data('output_gibbs.h5')
# gibbs_data.tau, gibbs_data.RB, etc., or alternatively gibbs_data[1000:2000]['RB'] etc.
emcee_data = simplebayesn.load_emcee_data('output_emcee.h5')
```

### Forward simulation

```python
true_params = {
    'tau': 0.15, 'RB': 3.26, 'M0_int': -19.48,
    'alpha': -0.16, 'beta_int': 0.0,
    'sigma_int2': 0.011, 'c0_int': -0.08,
    'alphac_int': -0.006, 'sigmac_int2': 0.038**2,
    'x0': -0.27, 'sigmax2': 1.07
}
sim = simplebayesn.simulators.sbsn.simulate_simplebayesn_salt_data_from_redshift_cov_arr(
    redshift, redshift_err, cov, true_params, seed=0
) # fixed vectors e.g. from ZTF data
```

### Selection correction (likelihood renormalisation)

When fitting data subject to colour or stretch cuts, pass selection limits to the emcee sampler to account for the truncation:

```python
simplebayesn.samplers.emcee_sampler(
    nwalkers, 1000, 10_000,
    iv_emcee, log_prior, salt_data,
    selection='mc',
    clim=(-0.3, 0.3), xlim=(-3, 3),
    num_sim_per_sample=2000,
    path='output_emcee_sel.h5'
) # GPU recommended
```

---

## Module structure

```
simplebayesn/
├── samplers/       # Gibbs and emcee samplers
├── simulators/     # Simple-BayeSN forward model
├── distributions/
│   ├── priors/     # Prior definitions for Gibbs and emcee
│   └── selection/  # Likelihood renormalisation for colour/stretch cuts
├── solvers/        # Maximum likelihood utilities
└── utils/
    ├── preprocessing.py  # SALT2 → SaltData conversion
    ├── data.py           # GibbsChainData / SaltData containers
    ├── initialize.py     # Initial value sampling
    ├── visualize.py      # Plotting utilities
    └── intrinsic.py      # Intrinsic population utilities
```

---

## Citation

If you use this code, please cite:

> Giunta, Karchev & Trotta (2026), *The colour variability of low-z SNe Ia is entirely explained by dust*

and the original Simple-BayeSN paper:

> Mandel, Scolnic, Shariff, Foley, and Kirshner, *The Type Ia Supernova Color–Magnitude Relation and Host Galaxy Dust: A Simple Hierarchical Bayesian Model*, ApJ, 842, 2, `https://doi.org/10.3847/1538-4357/aa6038`
