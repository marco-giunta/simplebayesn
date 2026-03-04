import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from corner import corner
import seaborn as sns
from ..utils.data import GibbsChainData
from matplotlib.animation import FuncAnimation
from matplotlib.colors import TABLEAU_COLORS as tab_colors
from matplotlib.patches import Patch

PARAMS_LATEX_MAP = {
    'M0_int': r'$M_0^{\text{int}}$',
    'alpha': r'$\alpha$',
    'beta_int': r'$\beta_{\text{int}}$',
    'c0_int':r'$c_0^{\text{int}}$',
    'alphac_int':r'$\alpha_c^{\text{int}}$',
    'x0':r'$x_0$',
    'sigma_int2':r'$\sigma_{\text{int}}^2$',
    'sigmac_int2':r'$(\sigma_c^{\text{int}})^2$',
    'sigmax2':r'$\sigma_x^2$',
    'RB':r'$R_B$',
    'tau':r'$\tau$'
}

def posterior_cornerplot(chain: GibbsChainData,
                         start_idx: int = 0, stop_idx: int = None,
                         title: str = None, levels = (0.393, 0.864),
                         show_joint_mean: bool = True,
                         truth_dict: dict = None,
                         show_marginal_mean: bool = False, show_marginal_std: bool = False,
                         show_datapoints: bool = False, show_titles: bool = True,
                         contours_color: str = 'dodgerblue', mean_color: str = 'darkblue', std_color: str = 'darkblue', truth_color: str =  'black',
                         axes_labels_fontsize = 25, diag_labels_fontsize = 18, ticks_labels_fontsize = 16, title_fontsize = 25,
                         params_to_plot: list = None, fig = None,
                         latex: bool = True,
                         *args, **kwargs):
    if params_to_plot is None:
        params_to_plot = ['tau', 'RB',
                          'x0', 'sigmax2',
                          'c0_int', 'alphac_int', 'sigmac_int2',
                          'M0_int', 'alpha', 'beta_int', 'sigma_int2']

    data = np.column_stack([chain[start_idx:stop_idx]['global_params'][k] for k in params_to_plot])
    means = data.mean(axis=0)
    stds = data.std(axis=0)

    if truth_dict is None:
        joint_dist_points = means if show_joint_mean else None
    else:
        joint_dist_points = np.array([truth_dict[p] for p in params_to_plot])

    fig = corner(
        data,
        levels = levels,
        plot_contours = True,
        fill_contours = True,
        show_titles = show_titles,
        plot_datapoints = show_datapoints,
        labels = [PARAMS_LATEX_MAP[k] if latex else k for k in params_to_plot],
        color = contours_color,
        label_kwargs = {'fontsize': axes_labels_fontsize},
        title_kwargs = {'fontsize': diag_labels_fontsize},
        weights = np.ones(data.shape[0]) / data.shape[0],
        truths = joint_dist_points,
        truth_color = mean_color if truth_dict is None else truth_color,
        fig = fig,
        *args, **kwargs
    )

    for ax in fig.get_axes():
        ax.tick_params(axis = 'both', labelsize = ticks_labels_fontsize)

    ndim = len(params_to_plot)
    axes = np.array(fig.axes).reshape((ndim, ndim))
    for i in range(ndim):
        ax = axes[i, i]
        if show_marginal_mean:
            ax.axvline(means[i], color=mean_color, lw=2)            # Truth line
        if show_marginal_std:
            ax.axvline(means[i] - stds[i], color=std_color, lw=1.5, ls="--")   # 1 sigma lower
            ax.axvline(means[i] + stds[i], color=std_color, lw=1.5, ls="--")   # 1 sigma upper
            ax.axvline(means[i] - 2*stds[i], color=std_color, lw=1.5, ls=":")  # 2 sigma lower
            ax.axvline(means[i] + 2*stds[i], color=std_color, lw=1.5, ls=":")  # 2 sigma upper

    if title is not None:
        fig.suptitle(title, fontsize = title_fontsize)

    return fig

def trace_plot(chain: GibbsChainData, param: str,
               start_idx: int = 0, stop_idx: int = None, title: str = None,
               show_mean: bool = True, show_std: bool = True,
               figsize = None, show_legend: bool = True,
               title_fontsize: int = 14, axes_labels_fontsize: int = None,
               legend_fontsize: int = None,
               ax = None, latex: bool = True):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    v = getattr(chain, param)[start_idx:stop_idx]
    ax.plot(v)

    mean = np.mean(v)
    std = np.std(v)
    if show_mean:
        ax.axhline(mean, color='red', linestyle='solid', linewidth=2, label=f'Mean: {mean:.5f}')
    if show_std:
        ax.axhline(mean - std, color='orange', linestyle='dashed', linewidth=2, label=f'$\\pm$1 Std Dev: [{(mean - std):.5f}, {(mean + std):.5f}]')
        ax.axhline(mean + std, color='orange', linestyle='dashed', linewidth=2)
        ax.axhline(mean - 2*std, color='green', linestyle='dotted', linewidth=2, label=f'$\\pm$2 Std Dev: [{(mean - 2*std):.5f}, {(mean + 2*std):.5f}]')
        ax.axhline(mean + 2*std, color='green', linestyle='dotted', linewidth=2)

    ax.set_xlabel('t', fontsize = axes_labels_fontsize)
    if title is None:
        ax.set_title(f'{PARAMS_LATEX_MAP[param] if latex else param}: {mean:.5f} $\\pm$ {std:.5f}', fontsize = title_fontsize)
    else:
        ax.set_title(title, fontsize = title_fontsize)

    if show_legend:
        ax.legend(fontsize = legend_fontsize)

    return ax

def marginal_posterior(chain: GibbsChainData, param: str,
                       start_idx: int = 0, stop_idx: int = None,
                       title: str = None,
                       kind: str = 'kde',
                       show_mean: bool = True, show_std: bool = True,
                       figsize = None, show_legend: bool = True,
                       title_fontsize: int = 18, axes_labels_fontsize: int = 14,
                       legend_fontsize: int = None,
                       ax = None, latex: bool = True,
                       *args, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    v = getattr(chain, param)[start_idx:stop_idx]

    if kind == 'kde':
        sns.kdeplot(v, ax = ax, *args, **kwargs)
    elif kind == 'hist':
        sns.histplot(v, ax = ax, stat = 'density', *args, **kwargs)
    else:
        raise ValueError(f'Invalid `kind=={kind}`')

    mean = np.mean(v)
    std = np.std(v)

    if show_mean:
        ax.axvline(mean, color = 'red', label = f'Mean: {mean:.5f}')
    if show_std:
        ax.axvline(mean - std, color='orange', linestyle='dashed', linewidth=2, label=f'$\\pm$1 Std Dev: [{(mean - std):.5f}, {(mean + std):.5f}]')
        ax.axvline(mean + std, color='orange', linestyle='dashed', linewidth=2)
        ax.axvline(mean - 2*std, color='green', linestyle='dotted', linewidth=2, label=f'$\\pm$2 Std Dev: [{(mean - 2*std):.5f}, {(mean + 2*std):.5f}]')
        ax.axvline(mean + 2*std, color='green', linestyle='dotted', linewidth=2)

    ax.set_xlabel(PARAMS_LATEX_MAP[param] if latex else param, fontsize = axes_labels_fontsize)
    if title is None:
        ax.set_title(PARAMS_LATEX_MAP[param] if latex else param, fontsize = title_fontsize)
    else:
        ax.set_title(title, fontsize = title_fontsize)

    if show_legend:
        ax.legend(fontsize = legend_fontsize)

    return ax

def intrinsic_magnitude_color_distribution_animation(chain_data: GibbsChainData,
                                                     start_idx: int = 0, stop_idx: int = None,
                                                     title: str = None,
                                                     step_stride: int = 500, color_dust: bool = True,
                                                     verbose: bool = False,
                                                     labels_fontsize = 12, title_fontsize = 14,
                                                     ticks_fontsize = 11, text_fontsize = 11):
    gp = chain_data[start_idx:stop_idx]['global_params']
    lp = chain_data[start_idx:stop_idx]['latent_params']

    c_int = lp['c_app'] - lp['E']
    M_int = lp['m_app'] - (lp['dist_mod'] + gp['RB'][:, np.newaxis] * lp['E'])
    M_int_ax = M_int  - gp['alpha'][:, np.newaxis] * lp['x']

    beta_int = gp['beta_int']

    num_iter = len(beta_int) # now different from chain_data.num_chain_samples due to slicing
    step_idx = np.arange(0, num_iter, step_stride)
    num_frames = len(step_idx)

    c_min, c_max = c_int.min(), c_int.max()
    M_min, M_max = M_int_ax.min(), M_int_ax.max()
    pad_c = 0.02 * (c_max - c_min) if c_max != c_min else 0.01
    pad_M = 0.02 * (M_max - M_min) if M_max != M_min else 0.1

    if color_dust:
        norm = plt.Normalize(vmin=lp['E'].min(), vmax=lp['E'].max())
        cmap = plt.cm.inferno

    fig, ax = plt.subplots(figsize=(6, 5))
    scat = ax.scatter([], [], s=10, alpha=0.6, color='k')
    line, = ax.plot([], [], lw=2)
    text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize = text_fontsize)

    ax.set_xlim(c_min - pad_c, c_max + pad_c)
    ax.set_ylim(M_max + pad_M, M_min - pad_M)  # reversed y-axis for magnitudes (bright up)
    ax.set_xlabel("Intrinsic color $c_{\\rm int}$" if verbose else '$c_{\\rm int}$',
                  fontsize = labels_fontsize)
    ax.set_ylabel("Stretch corrected intrinsic magnitude $M_{\\rm int}-\\alpha x$" if verbose else '$M_{\\rm int}-\\alpha x$',
                  fontsize = labels_fontsize)
    if title is None and verbose:
        ax.set_title("Evolution of intrinsic population and $\\beta_{\\rm int}$", fontsize = title_fontsize)
    else:
        ax.set_title(title, fontsize = title_fontsize)

    xvals = np.linspace(c_min - pad_c, c_max + pad_c, 200)

    def init():
        scat.set_offsets(np.empty((0, 2)))
        line.set_data([], [])
        text.set_text("")
        return scat, line, text

    def update(frame_i):
        step = step_idx[frame_i]
        b = beta_int[step]

        c = c_int[step, :]
        M = M_int_ax[step, :]
        offsets = np.column_stack([c, M])
        scat.set_offsets(offsets)
        if color_dust:
            scat.set_color(cmap(norm(lp['E'][step])))

        m0 = np.median(M)
        c0 = np.median(c)
        yvals = m0 + b * (xvals - c0)
        line.set_data(xvals, yvals)
        line.set_color("C0" if b > 0 else "C3")

        text.set_text(f"iter: {start_idx + step}\n" + "$\\beta_{\\rm int}$ = " + f"{b:+.3f}")
        return scat, line, text
    anim = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, interval=150)
    if color_dust:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(label='$E$ (dust reddening)' if verbose else '$E$',
                       fontsize = labels_fontsize)
        cbar.ax.tick_params(labelsize = ticks_fontsize)

    ax.tick_params(axis = 'both', labelsize = ticks_fontsize)

    plt.tight_layout()
    # plt.close(fig)

    return anim, fig

def extinguished_magnitude_color_distribution_animation(
    chain_data: GibbsChainData,
    start_idx: int = 0, stop_idx: int = None,
    title: str = None,
    step_stride: int = 500, color_dust: bool = True,
    verbose: bool = False,
    labels_fontsize = 12, title_fontsize = 14,
    ticks_fontsize = 11, text_fontsize = 11
):
    """
    Animate the evolution of the *extinguished* SN population:
    apparent color (c_app) vs. extinguished absolute magnitude (m_app - dist_mod),
    showing the slope governed by R_B at each MCMC iteration.
    """
    # --- extract sliced chain data
    gp = chain_data[start_idx:stop_idx]['global_params']
    lp = chain_data[start_idx:stop_idx]['latent_params']

    # --- key variables
    num_iter = len(gp['RB'])
    step_idx = np.arange(0, num_iter, step_stride)

    # apparent color and extinguished magnitude (distance-corrected)
    c_app = lp['c_app']                         # (n_iter, n_SN)
    M_ext = lp['m_app'] - lp['dist_mod']        # (n_iter, n_SN)
    M_ext_ax = M_ext - gp['alpha'][:, np.newaxis] * lp['x']

    RB = gp['RB']

    # --- define plot bounds with small padding
    c_min, c_max = c_app.min(), c_app.max()
    M_min, M_max = M_ext_ax.min(), M_ext_ax.max()
    pad_c = 0.02 * (c_max - c_min) if c_max != c_min else 0.01
    pad_M = 0.02 * (M_max - M_min) if M_max != M_min else 0.1

    # --- optional color coding by dust E
    if color_dust:
        norm = plt.Normalize(vmin=lp['E'].min(), vmax=lp['E'].max())
        cmap = plt.cm.inferno

    # --- figure setup
    fig, ax = plt.subplots(figsize=(6, 5))
    scat = ax.scatter([], [], s=10, alpha=0.6, color='k')
    line, = ax.plot([], [], lw=2)
    text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize = text_fontsize)

    ax.set_xlim(c_min - pad_c, c_max + pad_c)
    ax.set_ylim(M_max + pad_M, M_min - pad_M)  # reversed y-axis for magnitudes
    ax.set_xlabel("Apparent color $c_{\\rm app}$" if verbose else '$c_{\\rm app}$',
                  fontsize = labels_fontsize)
    ax.set_ylabel("Stretch-corrected extinguished magnitude $M_{\\rm ext} - \\alpha x$" if verbose else '$M_{\\rm ext} - \\alpha x$',
                  fontsize = labels_fontsize)
    if title is None and verbose:
        ax.set_title("Evolution of extinguished population and $R_B$", fontsize = title_fontsize)
    else:
        ax.set_title(title, fontsize = title_fontsize)

    xvals = np.linspace(c_min - pad_c, c_max + pad_c, 200)

    # --- init and update functions
    def init():
        scat.set_offsets(np.empty((0, 2)))
        line.set_data([], [])
        text.set_text("")
        return scat, line, text

    def update(frame_i):
        step = step_idx[frame_i]
        rB = RB[step]

        c = c_app[step, :]
        M = M_ext_ax[step, :]

        # scatter update
        offsets = np.column_stack([c, M])
        scat.set_offsets(offsets)

        if color_dust:
            scat.set_color(cmap(norm(lp['E'][step])))

        # slope line: fit around median
        m0 = np.median(M)
        c0 = np.median(c)
        yvals = m0 + rB * (xvals - c0)
        line.set_data(xvals, yvals)
        line.set_color("C0")

        # iteration label
        text.set_text(f"iter: {start_idx + step}\n$R_B$ = {rB:.2f}")

        return scat, line, text

    anim = FuncAnimation(fig, update, frames=len(step_idx),
                         init_func=init, blit=True, interval=150)
    if color_dust:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(label='$E$ (dust reddening)' if verbose else '$E$',
                       fontsize = labels_fontsize)
        cbar.ax.tick_params(labelsize = ticks_fontsize)

    ax.tick_params(axis = 'both', labelsize = ticks_fontsize)

    plt.tight_layout()
    return anim, fig

def compare_posterior_cornerplots(chains: list[GibbsChainData],
                                  start_idx: int = 0, stop_idx: int = None,
                                  title: str = None, levels = (0.393, 0.864),
                                  labels: list[str] = None,
                                  show_joint_mean: bool = True,
                                  truth_dict: dict = None,
                                  contours_colors: list[str] = None, mean_colors: list[str] = None,
                                  truth_color: str = 'black',
                                  axes_labels_fontsize = 25,
                                  ticks_labels_fontsize = 16, title_fontsize = 25,
                                  legend_fontsize: int = 20,
                                  params_to_plot: list = None,
                                  *args, **kwargs):
    
    if contours_colors is not None and len(contours_colors) != len(chains):
        raise ValueError(f'{len(chains) = } but {len(contours_colors) = }')
    if mean_colors is not None and len(mean_colors) != len(chains):
        raise ValueError(f'{len(chains) = } but {len(mean_colors) = }')

    if contours_colors is None:
        contours_colors = list(tab_colors.keys())[:len(chains)]
    if mean_colors is None:
        mean_colors = list(tab_colors.keys())[:len(chains)]

    shared_args = dict(start_idx = start_idx, stop_idx = stop_idx, title = title,
                       truth_dict = truth_dict, truth_color = truth_color,
                       levels = levels, show_joint_mean = show_joint_mean, show_marginal_mean = False,
                       show_marginal_std = False, show_titles = False, 
                       axes_labels_fontsize = axes_labels_fontsize,
                       ticks_labels_fontsize = ticks_labels_fontsize, title_fontsize = title_fontsize,
                       params_to_plot = params_to_plot)
    
    fig = posterior_cornerplot(chains[0], contours_color = contours_colors[0],
                               mean_color = mean_colors[0],
                               *args, **kwargs, **shared_args)

    for i in range(1, len(chains)):
        posterior_cornerplot(chains[i], contours_color = contours_colors[i],
                             mean_color = mean_colors[i],
                             *args, **kwargs, **shared_args, fig = fig)

    if labels is not None:
        legend_handles = [
            Patch(facecolor=contours_colors[i], label=labels[i])
            for i in range(len(chains))
        ]
        ndim = len(params_to_plot) if params_to_plot is not None else 11
        axes = np.array(fig.axes).reshape((ndim, ndim))
        legend_ax = axes[0, -1]
        legend_ax.legend(
            handles=legend_handles,
            fontsize=legend_fontsize,
            loc='upper right',
            frameon=True,
            framealpha=0.8,
        )

    return fig

def intrinsic_magnitude_color_distribution_frame(
    chain_data: GibbsChainData,
    iteration: int,
    start_idx: int = 0,
    stop_idx: int = None,
    title: str = None,
    color_dust: bool = True,
    verbose: bool = False,
    labels_fontsize = 12,
    title_fontsize = 14,
    ticks_fontsize = 11,
    text_fontsize = 11
):
    """
    Plot a single MCMC iteration of the intrinsic SN population:
    intrinsic color (c_int) vs. stretch-corrected intrinsic magnitude,
    including the beta_int slope line.
    """

    gp = chain_data[start_idx:stop_idx]['global_params']
    lp = chain_data[start_idx:stop_idx]['latent_params']

    c_int = lp['c_app'] - lp['E']
    M_int = lp['m_app'] - (lp['dist_mod'] + gp['RB'][:, np.newaxis] * lp['E'])
    M_int_ax = M_int - gp['alpha'][:, np.newaxis] * lp['x']

    beta_int = gp['beta_int']

    num_iter = len(beta_int)
    if iteration < 0 or iteration >= num_iter:
        raise IndexError(f"iteration must be in [0, {num_iter-1}]")

    c_min, c_max = c_int.min(), c_int.max()
    M_min, M_max = M_int_ax.min(), M_int_ax.max()
    pad_c = 0.02 * (c_max - c_min) if c_max != c_min else 0.01
    pad_M = 0.02 * (M_max - M_min) if M_max != M_min else 0.1

    c = c_int[iteration, :]
    M = M_int_ax[iteration, :]
    b = beta_int[iteration]

    fig, ax = plt.subplots(figsize=(6, 5))

    if color_dust:
        norm = plt.Normalize(vmin=lp['E'].min(), vmax=lp['E'].max())
        cmap = plt.cm.inferno
        scat = ax.scatter(
            c, M, s=10, alpha=0.6,
            c=lp['E'][iteration], cmap=cmap, norm=norm
        )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(label='$E$ (dust reddening)' if verbose else '$E$',
                       fontsize = labels_fontsize)
        cbar.ax.tick_params(labelsize = ticks_fontsize)
    else:
        scat = ax.scatter(c, M, s=10, alpha=0.6, color='k')

    # slope line centered at medians
    xvals = np.linspace(c_min - pad_c, c_max + pad_c, 200)
    m0 = np.median(M)
    c0 = np.median(c)
    yvals = m0 + b * (xvals - c0)

    ax.plot(
        xvals,
        yvals,
        lw=2,
        color="C0" if b > 0 else "C3"
    )

    ax.set_xlim(c_min - pad_c, c_max + pad_c)
    ax.set_ylim(M_max + pad_M, M_min - pad_M)  # reversed y-axis

    ax.set_xlabel(
        "Intrinsic color $c_{\\rm int}$" if verbose else "$c_{\\rm int}$",
        fontsize = labels_fontsize
    )
    ax.set_ylabel(
        "Stretch corrected intrinsic magnitude $M_{\\rm int}-\\alpha x$"
        if verbose else "$M_{\\rm int}-\\alpha x$",
        fontsize = labels_fontsize
    )

    if title is None and verbose:
        ax.set_title("Intrinsic population snapshot", fontsize = title_fontsize)
    else:
        ax.set_title(title, fontsize = title_fontsize)

    ax.text(
        0.02, 0.98,
        f"iter: {start_idx + iteration}\n"
        + "$\\beta_{\\rm int}$ = "
        + f"{b:+.3f}",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize = text_fontsize
    )

    ax.tick_params(axis = 'both', labelsize = ticks_fontsize)

    plt.tight_layout()
    return fig, ax

def extinguished_magnitude_color_distribution_frame(
    chain_data: GibbsChainData,
    iteration: int,
    start_idx: int = 0,
    stop_idx: int = None,
    title: str = None,
    color_dust: bool = False,
    verbose: bool = False,
    labels_fontsize = 12,
    title_fontsize = 14,
    ticks_fontsize = 11,
    text_fontsize = 11
):
    """
    Plot a single MCMC iteration of the extinguished SN population:
    apparent color (c_app) vs. stretch-corrected extinguished magnitude,
    including the R_B slope line.
    """

    gp = chain_data[start_idx:stop_idx]['global_params']
    lp = chain_data[start_idx:stop_idx]['latent_params']

    num_iter = len(gp['RB'])
    if iteration < 0 or iteration >= num_iter:
        raise IndexError(f"iteration must be in [0, {num_iter-1}]")

    c_app = lp['c_app']                         # (n_iter, n_SN)
    M_ext = lp['m_app'] - lp['dist_mod']        # (n_iter, n_SN)
    M_ext_ax = M_ext - gp['alpha'][:, np.newaxis] * lp['x']
    RB = gp['RB']

    c = c_app[iteration, :]
    M = M_ext_ax[iteration, :]
    rB = RB[iteration]

    # define plot bounds with small padding
    c_min, c_max = c_app.min(), c_app.max()
    M_min, M_max = M_ext_ax.min(), M_ext_ax.max()
    pad_c = 0.02 * (c_max - c_min) if c_max != c_min else 0.01
    pad_M = 0.02 * (M_max - M_min) if M_max != M_min else 0.1

    fig, ax = plt.subplots(figsize=(6, 5))

    if color_dust:
        norm = plt.Normalize(vmin=lp['E'].min(), vmax=lp['E'].max())
        cmap = plt.cm.inferno
        scat = ax.scatter(c, M, s=10, alpha=0.6,
                          c=lp['E'][iteration], cmap=cmap, norm=norm)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(label='$E$ (dust reddening)' if verbose else '$E$',
                       fontsize = labels_fontsize)
        cbar.ax.tick_params(labelsize = ticks_fontsize)
    else:
        scat = ax.scatter(c, M, s=10, alpha=0.6, color='k')

    # slope line centered at medians
    xvals = np.linspace(c_min - pad_c, c_max + pad_c, 200)
    m0 = np.median(M)
    c0 = np.median(c)
    yvals = m0 + rB * (xvals - c0)
    ax.plot(xvals, yvals, lw=2, color="C0")

    ax.set_xlim(c_min - pad_c, c_max + pad_c)
    ax.set_ylim(M_max + pad_M, M_min - pad_M)  # reversed y-axis
    ax.set_xlabel(
        "Apparent color $c_{\\rm app}$" if verbose else '$c_{\\rm app}$',
        fontsize = labels_fontsize
    )
    ax.set_ylabel(
        "Stretch-corrected extinguished magnitude $M_{\\rm ext} - \\alpha x$" if verbose else '$M_{\\rm ext} - \\alpha x$',
        fontsize = labels_fontsize
    )

    if title is None and verbose:
        ax.set_title("Extinguished population snapshot", fontsize = title_fontsize)
    else:
        ax.set_title(title, fontsize = title_fontsize)

    ax.text(
        0.02, 0.98,
        f"iter: {start_idx + iteration}\n$R_B$ = {rB:.2f}",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize = text_fontsize
    )

    ax.tick_params(axis = 'both', labelsize = ticks_fontsize)

    plt.tight_layout()
    return fig, ax
