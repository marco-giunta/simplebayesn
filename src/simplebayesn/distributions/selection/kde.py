import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import gaussian_kde
from scipy.integrate import dblquad

class KDEInterpolant:
    def __init__(self, c_sel, z_sel, c_com, z_com,
                 nc: int = 1000, nz: int = 1000, eps: float = 1e-8):
        cmin, cmax = np.min(c_com), np.max(c_com)
        zmin, zmax = np.min(z_com), np.max(z_com)

        kde_sel = gaussian_kde(np.vstack([c_sel, z_sel]))
        kde_com = gaussian_kde(np.vstack([c_com, z_com]))

        def sel_prob_unnnorm(c, z, eps=eps):
            cz = np.vstack([c, z])
            return kde_sel(cz) / (kde_com(cz) + eps)
        
        integral = dblquad(sel_prob_unnnorm, zmin, zmax, cmin, cmax)[0]

        def sel_prob(c, z, eps=eps):
            return sel_prob_unnnorm(c, z, eps) / integral
        
        c_vec = np.linspace(cmin, cmax, nc)
        z_vec = np.linspace(zmin, zmax, nz)

        c_grid, z_grid = np.meshgrid(c_vec, z_vec, indexing='ij')

        positions = np.vstack([c_grid.ravel(), z_grid.ravel()])
        p_grid = sel_prob(positions[0], positions[1]).T.reshape(c_grid.shape)
        
        self.c_vec = jnp.asarray(c_vec)
        self.z_vec = jnp.asarray(z_vec)
        self.p_grid = jnp.asarray(p_grid)

        self.nc = nc
        self.nz = nz
        self.eps = eps

    def __call__(self, c, z):
        return self.interpolate(c, z, self.c_vec, self.z_vec, self.p_grid, self.nc, self.nz)

    @staticmethod
    @jax.jit#(static_argnames=['nc', 'nz'])
    def interpolate(c, z, c_vec, z_vec, p_grid, nc, nz):
        outside = ((c < c_vec[0]) | (c > c_vec[-1]) |
                   (z < z_vec[0]) | (z > z_vec[-1]))


        c_idx = (c - c_vec[0]) / (c_vec[-1] - c_vec[0]) * (nc - 1)
        z_idx = (z - z_vec[0]) / (z_vec[-1] - z_vec[0]) * (nz - 1)

        c_idx = jnp.clip(c_idx, 0, nc - 1)
        z_idx = jnp.clip(z_idx, 0, nz - 1)

        c_i0 = jnp.floor(c_idx).astype(int)
        z_i0 = jnp.floor(z_idx).astype(int)
        c_i1 = jnp.minimum(c_i0 + 1, nc - 1)
        z_i1 = jnp.minimum(z_i0 + 1, nz - 1)

        c_frac = c_idx - c_i0
        z_frac = z_idx - z_i0

        val_00 = p_grid[c_i0, z_i0]
        val_01 = p_grid[c_i0, z_i1]
        val_10 = p_grid[c_i1, z_i0]
        val_11 = p_grid[c_i1, z_i1]

        val = (val_00 * (1 - c_frac) * (1 - z_frac) +
               val_01 * (1 - c_frac) * z_frac +
               val_10 * c_frac * (1 - z_frac) + 
               val_11 * c_frac * z_frac)
        
        return jnp.where(outside, 0., val)
    
def get_kde_interpolant(c_sel, z_sel, c_com, z_com,
                        nc=1000, nz=1000, eps=1e-8):
    return KDEInterpolant(c_sel, z_sel, c_com, z_com, nc, nz, eps)

# def get_kde_interpolant(c_sel, z_sel, c_com, z_com,
#                         nc = 1000, nz = 1000,
#                         eps=1e-8):
#     cmin, cmax = np.min(c_com), np.max(c_com)
#     zmin, zmax = np.min(z_com), np.max(z_com)

#     kde_sel = gaussian_kde(np.vstack([c_sel, z_sel]))
#     kde_com = gaussian_kde(np.vstack([c_com, z_com]))

#     def sel_prob_unnnorm(c, z, eps=eps):
#         cz = np.vstack([c, z])
#         return kde_sel(cz) / (kde_com(cz) + eps)
    
#     integral = dblquad(sel_prob_unnnorm, zmin, zmax, cmin, cmax)[0]

#     def sel_prob(c, z, eps=eps):
#         return sel_prob_unnnorm(c, z, eps) / integral
    
#     c_vec = np.linspace(cmin, cmax, nc)
#     z_vec = np.linspace(zmin, zmax, nz)

#     c_grid, z_grid = np.meshgrid(c_vec, z_vec, indexing='ij')

#     positions = np.vstack([c_grid.ravel(), z_grid.ravel()])
#     p_grid = sel_prob(positions[0], positions[1]).T.reshape(c_grid.shape)
    
#     c_vec = jnp.asarray(c_vec)
#     z_vec = jnp.asarray(z_vec)
#     p_grid = jnp.asarray(p_grid)

#     @jax.jit
#     def interpolant(c, z):
#         outside = ((c < c_vec[0]) | (c > c_vec[-1]) |
#                    (z < z_vec[0]) | (z > z_vec[-1]))


#         c_idx = (c - c_vec[0]) / (c_vec[-1] - c_vec[0]) * (nc - 1)
#         z_idx = (z - z_vec[0]) / (z_vec[-1] - z_vec[0]) * (nz - 1)

#         c_idx = jnp.clip(c_idx, 0, nc - 1)
#         z_idx = jnp.clip(z_idx, 0, nz - 1)

#         c_i0 = jnp.floor(c_idx).astype(int)
#         z_i0 = jnp.floor(z_idx).astype(int)
#         c_i1 = jnp.minimum(c_i0 + 1, nc - 1)
#         z_i1 = jnp.minimum(z_i0 + 1, nz - 1)

#         c_frac = c_idx - c_i0
#         z_frac = z_idx - z_i0

#         val_00 = p_grid[c_i0, z_i0]
#         val_01 = p_grid[c_i0, z_i1]
#         val_10 = p_grid[c_i1, z_i0]
#         val_11 = p_grid[c_i1, z_i1]

#         val = (val_00 * (1 - c_frac) * (1 - z_frac) +
#                val_01 * (1 - c_frac) * z_frac +
#                val_10 * c_frac * (1 - z_frac) + 
#                val_11 * c_frac * z_frac)
        
#         return jnp.where(outside, 0., val)

#     return interpolant