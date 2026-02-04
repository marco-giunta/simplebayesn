import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr, logsumexp
from functools import partial
from ...utils.data import SaltData, SaltDataCompact
from ...utils.intrinsic import get_mean_int, get_cov_int

def log_selection_probability_grid(global_params: dict,
                                   observed_data: SaltData | SaltDataCompact,
                                   mlim: tuple[float], clim: tuple[float], xlim: tuple[float],
                                   Nm: int, Nc: int, Nx: int):
    mean_int = get_mean_int(global_params).flatten()
    cov_int = get_cov_int(global_params)
    e1 = np.array([1, 0, 0])

    Sigma_inv = np.linalg.inv(
        cov_int + observed_data.cov +
        observed_data.sigma_mu_z2[:, np.newaxis, np.newaxis] * np.outer(e1, e1)
    )

    eE = np.array([global_params['RB'], 1, 0])
    sE = 1 / np.sqrt(
        np.einsum('i,nij,j->n', eE, Sigma_inv, eE)
    ) # (N_SN, )
    sE_exp = np.expand_dims(sE, axis=(1, 2, 3))
    del sE

    # np.array(np.meshgrid(...)) == np.stack(np.meshgrid(...), axis=0)
    N_SN = observed_data.num_samples
    volume_element = (np.diff(mlim) * np.diff(clim) * np.diff(xlim) / ((Nm - 1) * (Nc - 1) * (Nx - 1)))[0]

    d = np.repeat(
        np.stack(
            np.meshgrid(
                np.linspace(*mlim, num=Nm),
                np.linspace(*clim, num=Nc),
                np.linspace(*xlim, num=Nx)
            ),
            axis=0
        )[np.newaxis, ...],
        repeats=N_SN, axis=0
    ) # (N_SN, 3, Nm, Nc, Nx)

    dm = np.zeros_like(d)
    dm[:, 0] = observed_data.dist_mod[:, np.newaxis, np.newaxis, np.newaxis]
    # observed_data.dist_mod[:, np.newaxis, np.newaxis, np.newaxis] == np.expand_dims(observed_data.dist_mod, axis=(1, 2, 3))
    mi = mean_int[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
    # mean_int[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis] == np.expand_dims(mean_int, axis=(0, 2, 3))

    y = d - mi - dm # (N_SN, 3, Nm, Nc, Nx)
    del d, mi, dm

    E_hat = sE_exp ** 2 * \
          np.einsum('i,nij,njklm->nklm', eE, Sigma_inv, y)
    # (N_SN, Nm, Nc, Nx)

    log_prefactor = np.log(sE_exp * np.sqrt(2 * np.pi))

    EE = np.zeros_like(y)
    EE[:, 0] = global_params['RB'] * E_hat
    EE[:, 1] = E_hat
    v = y - EE
    del y, EE
    # log_norm_factor = -0.5 * (np.einsum(
    #     'niklm,nij,njklm->nklm',
    #     v, Sigma_inv, v
    # ) + np.log(np.expand_dims(
    #     ((2*np.pi)**3 / np.linalg.det(Sigma_inv)),
    #     axis=(1, 2, 3)
    # ))) # (N_SN, Nm, Nc, Nx)
    log_norm_factor = -0.5 * (np.einsum('niklm,nij,njklm->nklm', v, Sigma_inv, v)
                              + np.log(np.expand_dims(((2 * np.pi) ** 3 / np.linalg.det(Sigma_inv)), axis=(1, 2, 3))))
    del v

    log_exp_factor = -np.log(global_params['tau']) + (
        0.5 * \
            (sE_exp / global_params['tau']) ** 2 \
        - E_hat/global_params['tau']
    )
    log_cdf_factor = log_ndtr(
        E_hat / sE_exp \
        - sE_exp / global_params['tau']
    )
    del E_hat, sE_exp

    ll_grid = log_prefactor + log_norm_factor + log_exp_factor + log_cdf_factor
    del log_prefactor, log_norm_factor, log_exp_factor, log_cdf_factor
    
    max_ll = ll_grid.max(axis=(1, 2, 3))
    # integrals = (np.exp(ll_grid - max_ll[:, None, None, None]).sum(axis=(1, 2, 3)) * volume_element) * np.exp(max_ll)
    # return np.prod(integrals)
    log_integrals = logsumexp(ll_grid - max_ll[:, None, None, None], axis=(1, 2, 3)) + np.log(volume_element) + max_ll
    return log_integrals.sum()

def preprocess_arguments_log_selection_probability_grid_jax(global_params: dict,
                                                            observed_data: SaltData | SaltDataCompact):
    return {
        'RB': global_params['RB'],
        'tau': global_params['tau'],
        'mean_int': jnp.asarray(get_mean_int(global_params)),
        'cov_int': jnp.asarray(get_cov_int(global_params)),
        'observed_data_cov': jnp.asarray(observed_data.cov),
        'observed_data_sigma_mu_z2': jnp.asarray(observed_data.sigma_mu_z2),
        'observed_data_num_samples': observed_data.num_samples,
        'observed_data_dist_mod': jnp.asarray(observed_data.dist_mod),
    }

# @jax.jit
# def logsumexp_stable(a: jax.Array, b: jax.Array = None, axis = None):
#     if b is not None:
#         m = jnp.maximum(a, b)
#         return m + jnp.logaddexp(a - m, b - m)
#     m = jnp.max(a, axis=axis)
#     return m + logsumexp(a - m, axis=axis)

@partial(jax.jit, static_argnames=('axis',))
def logsumexp_stable(a: jax.Array, b: jax.Array = None, axis = None):
    if b is not None:
        m = jnp.maximum(a, b)
        return m + jnp.logaddexp(a - m, b - m)
    m = jnp.max(a, axis=axis, keepdims=True)
    return jnp.squeeze(m, axis=axis) + logsumexp(a - m, axis=axis)

@partial(jax.jit, static_argnames = [
    'mlim', 'clim', 'xlim', 'Nm', 'Nc', 'Nx',
    'observed_data_num_samples'
])
def log_selection_probability_grid_jax_2(mlim: tuple[float], clim: tuple[float], xlim: tuple[float],
                                         Nm: int, Nc: int, Nx: int,
                                         tau, RB, mean_int, cov_int,
                                         observed_data_cov, observed_data_sigma_mu_z2,
                                         observed_data_num_samples, observed_data_dist_mod):
    mean_int = mean_int.flatten()
    e1 = jnp.array([1, 0, 0])
    eE = jnp.array([RB, 1, 0])

    Sigma_inv = jnp.linalg.inv(
        cov_int + observed_data_cov + observed_data_sigma_mu_z2[:, None, None] * jnp.outer(e1, e1)
    )

    sE = 1 / jnp.sqrt(jnp.einsum('i,nij,j->n',eE, Sigma_inv, eE))
    m_vals = jnp.linspace(mlim[0], mlim[1], Nm)
    c_vals = jnp.linspace(clim[0], clim[1], Nc)
    x_vals = jnp.linspace(xlim[0], xlim[1], Nx)

    c_grid, x_grid = jnp.meshgrid(c_vals, x_vals, indexing='ij')  # (Nc, Nx)

    y1_grid = c_grid - mean_int[1]
    y2_grid = x_grid - mean_int[2]
    del c_grid, x_grid, c_vals, x_vals

    log_volume_element = jnp.log(
        jnp.diff(jnp.array(mlim)) * 
        jnp.diff(jnp.array(clim)) * 
        jnp.diff(jnp.array(xlim)) / ((Nm - 1) * (Nc - 1) * (Nx - 1))
    )[0]
    
    def m_step(m_carry, m_val, Sigma_inv_i, sE_i, dist_mod_i):
        y0_grid = jnp.full_like(y1_grid, m_val - mean_int[0] - dist_mod_i)
        y = jnp.stack([y0_grid, y1_grid, y2_grid], axis=0)
        E_hat = sE_i ** 2 * jnp.einsum('i,ij,jkl->kl', eE, Sigma_inv_i, y)
        del y
        log_prefactor = jnp.log(sE_i * jnp.sqrt(2 * jnp.pi))

        v = jnp.stack([
            y0_grid - RB * E_hat,
            y1_grid - E_hat,
            y2_grid
        ], axis=0)

        _, logdet = jnp.linalg.slogdet(Sigma_inv_i)
        log_norm_factor = -0.5 * (jnp.einsum('ikl,ij,jkl->kl', v, Sigma_inv_i, v) \
                                  + jnp.log((2 * jnp.pi) ** 3) - logdet)
        del v
        log_exp_factor = -jnp.log(tau) + (0.5 * (sE_i / tau) ** 2 - E_hat / tau)
        log_cdf_factor = log_ndtr(E_hat / sE_i - sE_i / tau)

        ll_grid = log_prefactor + log_norm_factor + log_exp_factor + log_cdf_factor
        ll_integral = logsumexp_stable(ll_grid)

        new_m_carry = jnp.where(jnp.isneginf(m_carry),
                                ll_integral,
                                logsumexp_stable(m_carry, ll_integral))
        return new_m_carry, None
    
    def per_sample_step(carry, i):
        m_init = -jnp.inf
        m_final, _ = jax.lax.scan(
            lambda carry, m_current: m_step(carry, m_current, Sigma_inv[i], sE[i], observed_data_dist_mod[i]),
            m_init,
            m_vals
        )
        new_carry = m_final + log_volume_element + carry
        return new_carry, None

    return jax.lax.scan(per_sample_step, jnp.array(0), jnp.arange(observed_data_num_samples))[0]

@partial(jax.jit, static_argnames = [
    'mlim', 'clim', 'xlim', 'Nm', 'Nc', 'Nx',
    'observed_data_num_samples'
])
def log_selection_probability_grid_jax_3(mlim: tuple[float], clim: tuple[float], xlim: tuple[float],
                                         Nm: int, Nc: int, Nx: int,
                                         tau, RB, mean_int, cov_int,
                                         observed_data_cov, observed_data_sigma_mu_z2,
                                         observed_data_num_samples, observed_data_dist_mod):
    mean_int = mean_int.flatten()
    e1 = jnp.array([1, 0, 0])
    eE = jnp.array([RB, 1, 0])

    Sigma_inv = jnp.linalg.inv(
        cov_int + observed_data_cov + observed_data_sigma_mu_z2[:, None, None] * jnp.outer(e1, e1)
    )

    sE = 1 / jnp.sqrt(
        jnp.einsum('i,nij,j->n', eE, Sigma_inv, eE)
    )

    d = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(*mlim, num=Nm),
            jnp.linspace(*clim, num=Nc),
            jnp.linspace(*xlim, num=Nx),
            indexing='ij'
        ),
        axis=0
    )

    y_base = d - mean_int[:, None, None, None]
    del d, mean_int

    log_volume_element = jnp.log(
        jnp.diff(jnp.array(mlim)) * \
        jnp.diff(jnp.array(clim)) * \
        jnp.diff(jnp.array(xlim)) / \
        ((Nm - 1) * (Nc - 1) * (Nx - 1))
    )[0]

    def per_sample(i, carry):
        y = y_base - jnp.zeros_like(y_base).at[0].add(observed_data_dist_mod[i])

        E_hat = (sE[i] ** 2) * jnp.einsum('i,ij,jklm->klm', eE, Sigma_inv[i], y)
        log_prefactor = jnp.log(sE[i] * jnp.sqrt(2 * jnp.pi))

        v = y - jnp.zeros_like(y).at[0].set(RB * E_hat).at[1].set(E_hat)
        del y

        log_norm_factor = -0.5 * (
            jnp.einsum('iklm,ij,jklm->klm', v, Sigma_inv[i], v) + \
            3 * jnp.log(2 * jnp.pi) - jnp.linalg.slogdet(Sigma_inv[i])[1]
        )
        del v

        log_exp_factor = -jnp.log(tau) + (0.5 * (sE[i] / tau) ** 2 - E_hat / tau)

        log_cdf_factor = log_ndtr(E_hat / sE[i] - sE[i] / tau)
        del E_hat

        ll_grid = log_prefactor + log_norm_factor + log_exp_factor + log_cdf_factor
        del log_prefactor, log_norm_factor, log_exp_factor, log_cdf_factor

        ll_integral = logsumexp_stable(ll_grid) + log_volume_element
        return ll_integral + carry

    return jax.lax.fori_loop(0, observed_data_num_samples, per_sample, 0)

@partial(jax.jit, static_argnames = [
    'mlim', 'clim', 'xlim', 'Nm', 'Nc', 'Nx',
    'observed_data_num_samples'
])
def log_selection_probability_grid_jax_4(mlim: tuple[float], clim: tuple[float], xlim: tuple[float],
                                         Nm: int, Nc: int, Nx: int,
                                         tau, RB, mean_int, cov_int,
                                         observed_data_cov, observed_data_sigma_mu_z2,
                                         observed_data_num_samples, observed_data_dist_mod):
    mean_int = mean_int.flatten()
    e1 = jnp.array([1, 0, 0])

    Sigma_inv = jnp.linalg.inv(
        cov_int + observed_data_cov +
        observed_data_sigma_mu_z2[:, None, None] * jnp.outer(e1, e1)
    )

    eE = jnp.array([RB, 1, 0])
    sE = 1 / jnp.sqrt(
        jnp.einsum('i,nij,j->n', eE, Sigma_inv, eE)
    )  # shape (N_SN, )
    sE_exp = sE[:, None, None, None]
    del sE

    N_SN = observed_data_num_samples

    log_volume_element = jnp.log(
        jnp.diff(jnp.array(mlim)) * \
        jnp.diff(jnp.array(clim)) * \
        jnp.diff(jnp.array(xlim)) / \
        ((Nm - 1) * (Nc - 1) * (Nx - 1))
    )[0]

    grid_mesh = jnp.meshgrid(
        jnp.linspace(mlim[0], mlim[1], Nm),
        jnp.linspace(clim[0], clim[1], Nc),
        jnp.linspace(xlim[0], xlim[1], Nx),
        indexing='ij'
    )

    d = jnp.stack(grid_mesh, axis=0)  # shape (3, Nm, Nc, Nx)
    d = jnp.broadcast_to(d, (N_SN, 3, Nm, Nc, Nx)) # cfr repeat
    del grid_mesh

    dm = jnp.zeros_like(d)
    dm = dm.at[:, 0].set(observed_data_dist_mod[:, None, None, None])

    mi = jnp.expand_dims(mean_int, axis=(0, 2, 3, 4))

    y = d - mi - dm  # shape (N_SN, 3, Nm, Nc, Nx)
    del d, mi, dm

    E_hat = sE_exp ** 2 * jnp.einsum('i,nij,njklm->nklm', eE, Sigma_inv, y)  # shape (N_SN, Nm, Nc, Nx)

    log_prefactor = jnp.log(sE_exp * jnp.sqrt(2 * jnp.pi))

    EE = jnp.zeros_like(y)
    EE = EE.at[:, 0].set(RB * E_hat)
    EE = EE.at[:, 1].set(E_hat)

    v = y - EE
    del y, EE

    log_norm_factor = -0.5 * (jnp.einsum('niklm,nij,njklm->nklm', v, Sigma_inv, v)
                              + jnp.log(jnp.expand_dims(((2 * jnp.pi) ** 3 / jnp.linalg.det(Sigma_inv)), axis=(1, 2, 3))))
    del v

    log_exp_factor = -jnp.log(tau) + (0.5 * (sE_exp / tau) ** 2 - E_hat / tau)

    log_cdf_factor = log_ndtr(E_hat / sE_exp - sE_exp / tau)
    del E_hat, sE_exp

    ll_grid = log_prefactor + log_norm_factor + log_exp_factor + log_cdf_factor
    del log_prefactor, log_norm_factor, log_exp_factor, log_cdf_factor

    # max_ll = ll_grid.max(axis=(1, 2, 3))
    # log_integrals = logsumexp(ll_grid - max_ll[:, None, None, None], axis=(1, 2, 3)) + log_volume_element + max_ll

    log_integrals = logsumexp_stable(ll_grid, axis=(1, 2, 3)) + log_volume_element
    return log_integrals.sum()

@partial(jax.jit, static_argnames = [
    'mlim', 'clim', 'xlim', 'Nm', 'Nc', 'Nx',
    'observed_data_num_samples', 'batch_size'
])
def log_selection_probability_grid_jax_4b(mlim: tuple[float], clim: tuple[float], xlim: tuple[float],
                                          Nm: int, Nc: int, Nx: int,
                                          tau, RB, mean_int, cov_int,
                                          observed_data_cov, observed_data_sigma_mu_z2,
                                          observed_data_num_samples, observed_data_dist_mod,
                                          batch_size: int):
    mean_int = mean_int.flatten()
    mean_int = jnp.expand_dims(mean_int, axis=(0, 2, 3, 4))
    e1 = jnp.array([1, 0, 0])
    eE = jnp.array([RB, 1, 0])

    log_volume_element = jnp.log(
        jnp.diff(jnp.array(mlim)) * \
        jnp.diff(jnp.array(clim)) * \
        jnp.diff(jnp.array(xlim)) / \
        ((Nm - 1) * (Nc - 1) * (Nx - 1))
    )[0]

    d = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(mlim[0], mlim[1], Nm),
            jnp.linspace(clim[0], clim[1], Nc),
            jnp.linspace(xlim[0], xlim[1], Nx),
            indexing='ij'
        ),
        axis=0
    )[None, ...]

    def process_batch(start_idx):
        observed_data_cov_batch = jax.lax.dynamic_slice_in_dim(observed_data_cov, start_idx, batch_size)
        observed_data_sigma_mu_z2_batch = jax.lax.dynamic_slice_in_dim(observed_data_sigma_mu_z2, start_idx, batch_size)
        observed_data_dist_mod_batch = jax.lax.dynamic_slice_in_dim(observed_data_dist_mod, start_idx, batch_size)

        # current_batch_size = len(observed_data_sigma_mu_z2_batch)
        idxs = start_idx + jnp.arange(batch_size)
        valid_mask = idxs < observed_data_num_samples # slice in dim: clipping in start_idx, non stop_idx

        Sigma_inv = jnp.linalg.inv(
            cov_int + observed_data_cov_batch +
            observed_data_sigma_mu_z2_batch[:, None, None] * jnp.outer(e1, e1)
        )

        sE = 1 / jnp.sqrt(jnp.einsum('i,nij,j->n', eE, Sigma_inv, eE))[:, None, None, None]

        y = jnp.zeros((batch_size, 3, Nm, Nc, Nx))\
              .at[:, 0].add(-observed_data_dist_mod_batch[:, None, None, None]) \
              + d - mean_int

        E_hat = sE ** 2 * jnp.einsum('i,nij,njklm->nklm', eE, Sigma_inv, y)
        log_prefactor = jnp.log(sE * jnp.sqrt(2 * jnp.pi))

        v = y.at[:, 0].add(-RB * E_hat).at[:, 1].add(-E_hat)
        del y

        log_norm_factor = -0.5 * (
            jnp.einsum('niklm,nij,njklm->nklm', v, Sigma_inv, v)
            # + jnp.log(((2 * jnp.pi) ** 3 / jnp.linalg.det(Sigma_inv))[:, None, None, None])
            + (3 * jnp.log(2 * jnp.pi) - jnp.linalg.slogdet(Sigma_inv)[1])[:, None, None, None]
        )
        del v

        log_exp_factor = -jnp.log(tau) + (0.5 * (sE / tau) ** 2 - E_hat / tau)
        log_cdf_factor = log_ndtr(E_hat / sE - sE / tau)
        del E_hat

        ll_grid = log_prefactor + log_norm_factor + log_exp_factor + log_cdf_factor
        del log_prefactor, log_norm_factor, log_exp_factor, log_cdf_factor
        # max_ll = ll_grid.max(axis=(1, 2, 3))
        # log_integrals = (
        #     logsumexp(ll_grid - max_ll[:, None, None, None], axis=(1, 2, 3))
        #     + log_volume_element
        #     + max_ll
        # )
        log_integrals = logsumexp_stable(ll_grid, axis=(1, 2, 3)) + log_volume_element
        return jnp.where(valid_mask, log_integrals, 0.0).sum()


    # Loop with lax.fori_loop for JIT compatibility
    def loop_body(i, acc):
        return acc + process_batch(i * batch_size)

    num_batches = (observed_data_num_samples + batch_size - 1) // batch_size
    total_log_integral = jax.lax.fori_loop(0, num_batches, loop_body, 0.0)
    return total_log_integral
