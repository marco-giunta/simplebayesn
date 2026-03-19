import numpy as np
import matplotlib.pyplot as plt
from corner import corner
import seaborn as sns
from ..utils.data import GibbsChainData
from matplotlib.animation import FuncAnimation
from matplotlib.colors import TABLEAU_COLORS as tab_colors
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

PARAMS_LATEX_MAP = {
    'M0_int': r'$M_0^{\text{int}}$',
    'alpha': r'$\alpha$',
    'beta_int': r'$\beta_{\text{int}}$',
    'c0_int':r'$c_0^{\text{int}}$',
    'alphac_int':r'$\alpha_c^{\text{int}}$',
    'x0':r'$x_0$',
    'sigma_int2':r'$\sigma_{\text{int}}^2$',
    'sigmac_int2':r'$\sigma_{c, \text{int}}^2$',
    'sigmax2':r'$\sigma_x^2$',
    'RB':r'$R_B$',
    'tau':r'$\tau$',
    'sigma_int':r'$\sigma_{\text{int}}$',
    'sigmac_int':r'$\sigma_{c, \text{int}}$',
    'sigmax':r'$\sigma_x$',
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
                          'x0', 'sigmax',
                          'c0_int', 'alphac_int', 'sigmac_int',
                          'M0_int', 'alpha', 'beta_int', 'sigma_int']

    data = np.column_stack([chain[k][start_idx:stop_idx] for k in params_to_plot])
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

def intrinsic_magnitude_color_distribution_animation(chain: GibbsChainData,
                                                     start_idx: int = 0, stop_idx: int = None,
                                                     title: str = None,
                                                     step_stride: int = 500, color_dust: bool = True,
                                                     verbose: bool = False,
                                                     labels_fontsize = 12, title_fontsize = 14,
                                                     ticks_fontsize = 11, text_fontsize = 11,
                                                     figsize = (6, 5)):
    params = chain[start_idx:stop_idx]

    c_int = params['c_app'] - params['E']
    M_int = params['m_app'] - (params['dist_mod'] + params['RB'][:, np.newaxis] * params['E'])
    M_int_ax = M_int  - params['alpha'][:, np.newaxis] * params['x']

    beta_int = params['beta_int']

    num_iter = len(beta_int) # now different from chain.num_chain_samples due to slicing
    step_idx = np.arange(0, num_iter, step_stride)
    num_frames = len(step_idx)

    c_min, c_max = c_int.min(), c_int.max()
    M_min, M_max = M_int_ax.min(), M_int_ax.max()
    pad_c = 0.02 * (c_max - c_min) if c_max != c_min else 0.01
    pad_M = 0.02 * (M_max - M_min) if M_max != M_min else 0.1

    if color_dust:
        norm = plt.Normalize(vmin=params['E'].min(), vmax=params['E'].max())
        cmap = plt.cm.inferno

    fig, ax = plt.subplots(figsize=figsize)
    scat = ax.scatter([], [], s=10, alpha=0.6, color='k')
    line, = ax.plot([], [], lw=2)
    text = ax.text(0.75, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize = text_fontsize)

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
            scat.set_color(cmap(norm(params['E'][step])))

        m0 = np.median(M)
        c0 = np.median(c)
        yvals = m0 + b * (xvals - c0)
        line.set_data(xvals, yvals)
        line.set_color("C0" if b > 0 else "C3")

        text.set_text(f"iter: {start_idx + step}\n" + "$\\beta_{\\rm int}$ = " + f"{b:+.3f}",
                      color="C0" if b > 0 else "C3")
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
    chain: GibbsChainData,
    start_idx: int = 0, stop_idx: int = None,
    title: str = None,
    step_stride: int = 500, color_dust: bool = True,
    verbose: bool = False,
    labels_fontsize = 12, title_fontsize = 14,
    ticks_fontsize = 11, text_fontsize = 11,
    figsize = (6, 5)
):
    """
    Animate the evolution of the *extinguished* SN population:
    apparent color (c_app) vs. extinguished absolute magnitude (m_app - dist_mod),
    showing the slope governed by R_B at each MCMC iteration.
    """
    # --- extract sliced chain data
    params = chain[start_idx:stop_idx]

    # --- key variables
    num_iter = len(params['RB'])
    step_idx = np.arange(0, num_iter, step_stride)

    # apparent color and extinguished magnitude (distance-corrected)
    c_app = params['c_app']                         # (n_iter, n_SN)
    M_ext = params['m_app'] - params['dist_mod']    # (n_iter, n_SN)
    M_ext_ax = M_ext - params['alpha'][:, np.newaxis] * params['x']

    RB = params['RB']

    # --- define plot bounds with small padding
    c_min, c_max = c_app.min(), c_app.max()
    M_min, M_max = M_ext_ax.min(), M_ext_ax.max()
    pad_c = 0.02 * (c_max - c_min) if c_max != c_min else 0.01
    pad_M = 0.02 * (M_max - M_min) if M_max != M_min else 0.1

    # --- optional color coding by dust E
    if color_dust:
        norm = plt.Normalize(vmin=params['E'].min(), vmax=params['E'].max())
        cmap = plt.cm.inferno

    # --- figure setup
    fig, ax = plt.subplots(figsize=figsize)
    scat = ax.scatter([], [], s=10, alpha=0.6, color='k')
    line, = ax.plot([], [], lw=2)
    text = ax.text(0.75, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize = text_fontsize)

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
            scat.set_color(cmap(norm(params['E'][step])))

        # slope line: fit around median
        m0 = np.median(M)
        c0 = np.median(c)
        yvals = m0 + rB * (xvals - c0)
        line.set_data(xvals, yvals)
        line.set_color("C0")

        # iteration label
        text.set_text(f"iter: {start_idx + step}\n$R_B$ = {rB:.2f}", color="C0")

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
                                  legend_fontsize: int = 20, show_sn_num: bool = True,
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
        if show_sn_num:
            labels = [l + f' ({c.num_data_samples} SNe)' for c, l in zip(chains, labels)]
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
    chain: GibbsChainData,
    iteration: int,
    start_idx: int = 0,
    stop_idx: int = None,
    title: str = None,
    color_dust: bool = True,
    verbose: bool = False,
    labels_fontsize = 12,
    title_fontsize = 14,
    ticks_fontsize = 11,
    text_fontsize = 11,
    figsize = (6, 5)
):
    """
    Plot a single MCMC iteration of the intrinsic SN population:
    intrinsic color (c_int) vs. stretch-corrected intrinsic magnitude,
    including the beta_int slope line.
    """

    params = chain[start_idx:stop_idx]

    c_int = params['c_app'] - params['E']
    M_int = params['m_app'] - (params['dist_mod'] + params['RB'][:, np.newaxis] * params['E'])
    M_int_ax = M_int - params['alpha'][:, np.newaxis] * params['x']

    beta_int = params['beta_int']

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

    fig, ax = plt.subplots(figsize=figsize)

    if color_dust:
        norm = plt.Normalize(vmin=params['E'].min(), vmax=params['E'].max())
        cmap = plt.cm.inferno
        scat = ax.scatter(
            c, M, s=10, alpha=0.6,
            c=params['E'][iteration], cmap=cmap, norm=norm
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
        0.75, 0.98,
        f"iter: {start_idx + iteration}\n"
        + "$\\beta_{\\rm int}$ = "
        + f"{b:+.3f}",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize = text_fontsize,
        color="C0" if b > 0 else "C3"
    )

    ax.tick_params(axis = 'both', labelsize = ticks_fontsize)

    plt.tight_layout()
    return fig, ax

def extinguished_magnitude_color_distribution_frame(
    chain: GibbsChainData,
    iteration: int,
    start_idx: int = 0,
    stop_idx: int = None,
    title: str = None,
    color_dust: bool = True,
    verbose: bool = False,
    labels_fontsize = 12,
    title_fontsize = 14,
    ticks_fontsize = 11,
    text_fontsize = 11,
    figsize = (6, 5)
):
    """
    Plot a single MCMC iteration of the extinguished SN population:
    apparent color (c_app) vs. stretch-corrected extinguished magnitude,
    including the R_B slope line.
    """

    params = chain[start_idx:stop_idx]

    num_iter = len(params['RB'])
    if iteration < 0 or iteration >= num_iter:
        raise IndexError(f"iteration must be in [0, {num_iter-1}]")

    c_app = params['c_app']                         # (n_iter, n_SN)
    M_ext = params['m_app'] - params['dist_mod']    # (n_iter, n_SN)
    M_ext_ax = M_ext - params['alpha'][:, np.newaxis] * params['x']
    RB = params['RB']

    c = c_app[iteration, :]
    M = M_ext_ax[iteration, :]
    rB = RB[iteration]

    # define plot bounds with small padding
    c_min, c_max = c_app.min(), c_app.max()
    M_min, M_max = M_ext_ax.min(), M_ext_ax.max()
    pad_c = 0.02 * (c_max - c_min) if c_max != c_min else 0.01
    pad_M = 0.02 * (M_max - M_min) if M_max != M_min else 0.1

    fig, ax = plt.subplots(figsize=figsize)

    if color_dust:
        norm = plt.Normalize(vmin=params['E'].min(), vmax=params['E'].max())
        cmap = plt.cm.inferno
        scat = ax.scatter(c, M, s=10, alpha=0.6,
                          c=params['E'][iteration], cmap=cmap, norm=norm)
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
        0.75, 0.98,
        f"iter: {start_idx + iteration}\n$R_B$ = {rB:.2f}",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize = text_fontsize,
        color="C0"
    )

    ax.tick_params(axis = 'both', labelsize = ticks_fontsize)

    plt.tight_layout()
    return fig, ax

def plot_latent_bias(chain: GibbsChainData,
                     host_vec: np.ndarray, host_vec_err: np.ndarray = None,
                     xlabel: str = None, xval: float = None,
                     color_vec: np.ndarray = None, color_vec_split_value: float = None, clabel: str = None,
                     start_idx: int = 0, stop_idx: int = None,
                     x_min: float = None, x_max: float = None,
                     markersize = 15, pop1_color: str = '#3778bf', pop2_color: str = '#e05c3a',
                     n_bins_trend: int = 10, trend_color: str = "#1E1E1E", bin_capsize = 3, bin_markersize = 5,
                     n_bins_hist: int = 20, hist_color: str = '#aaaaaa', hist_edge_color: str = '#666666',
                     hline_color: str = '#444444', vline_color: str = '#444444',
                     extra_hlines: dict = None, mass_step_labels_loc: str = None,
                     legend_fontsize: int = 10,
                     show_kde: bool = True,
                     figsize = (10,15)):

    def add_binned_trend(ax, x, y, yerr):
        x, y, yerr = np.asarray(x), np.asarray(y), np.asarray(yerr)

        bins = np.linspace(x.min(), x.max(), n_bins_trend + 1)
        bin_centers, bin_means, bin_errs = np.zeros(n_bins_trend), np.zeros(n_bins_trend), np.zeros(n_bins_trend)

        for i in range(n_bins_trend):
            mask = (x >= bins[i]) & (x < bins[i+1])
            if mask.sum() < 2:
                continue # at least 2 points per bin to compute var
            w = 1 / yerr[mask]**2
            mean = np.average(y[mask], weights=w)

            # standard error^2 of weighted mean
            var_mean = 1 / w.sum() # sum of 1/var
            # weighted var of the points around the mean
            var_weighted = np.average((y[mask] - mean)**2, weights=w)
            # combined: intrinsic data point scatter + uncertainty on the mean
            err = np.sqrt(var_mean + var_weighted)

            bin_centers[i] = (bins[i] + bins[i+1]) / 2
            bin_means[i] = mean
            bin_errs[i] = err

        ax.errorbar(bin_centers, bin_means, yerr = bin_errs,
                    fmt = 's-', color = trend_color, capsize = bin_capsize,
                    markersize = bin_markersize, lw = 1.5, zorder = 5)

    def add_kde_marginal(ax, y, yerr):
        y = np.asarray(y)
        yerr = np.asarray(yerr)
        y_grid = np.linspace(y.min(), y.max(), 300)

        if color_vec is not None and color_vec_split_value is not None:
            color_mask = np.asarray(color_vec) <= color_vec_split_value
            for mask, color in [(color_mask, pop1_color), (~color_mask, pop2_color)]:
                if mask.sum() > 1:
                    kde = gaussian_kde(y[mask], weights=1/yerr[mask]**2)
                    ax.fill_betweenx(y_grid, kde(y_grid), alpha=0.25, color=color)
                    ax.plot(kde(y_grid), y_grid, color=color, lw=1.2)
        else:
            kde = gaussian_kde(y, weights=1/yerr**2)
            ax.fill_betweenx(y_grid, kde(y_grid), alpha=0.3, color=pop1_color)
            ax.plot(kde(y_grid), y_grid, color=pop1_color, lw=1.2)

        ax.set_xticks([])
        ax.tick_params(axis='y', left=False, right=False, labelleft=False, labelright=False)
        

    def plot_points(ax, y, yerr):
        # errorbars first (behind markers)
        ax.errorbar(host_vec, y, xerr=host_vec_err, yerr=yerr,
                    fmt='none', ecolor='grey', alpha=0.3, zorder=1)
        # points with color scattered on top of point-less grey errorbars
        if color_vec is not None and color_vec_split_value is not None:
            color_mask = color_vec <= color_vec_split_value
            ax.scatter(host_vec[color_mask], y[color_mask], c = pop1_color,
                       s = markersize, alpha = 0.7, zorder = 2)
            ax.scatter(host_vec[~color_mask], y[~color_mask], c=pop2_color,
                       s = markersize, alpha = 0.7, zorder = 2)
        else:
            ax.scatter(host_vec, y, c = pop1_color, s = markersize, alpha = 0.7, zorder = 2)
        add_binned_trend(ax, host_vec, y, yerr)

    if host_vec_err is None:
        host_vec_err = np.zeros_like(host_vec)

    if xval == 'median':
        xval = np.median(host_vec)
    elif xval == 'mean':
        xval = np.mean(host_vec)

    a_x = chain.alpha[start_idx:stop_idx][:, None] * chain.x[start_idx:stop_idx]
    alpha_x = np.mean(a_x, axis = 0)
    alpha_x_err = np.std(a_x, axis = 0)

    b_c = chain.beta_int[start_idx:stop_idx][:, None] * (chain.c_app - chain.E)[start_idx:stop_idx]
    beta_int_c = np.mean(b_c, axis = 0)
    beta_int_c_err = np.std(b_c, axis = 0)

    r_e = chain.RB[start_idx:stop_idx][:, None] * chain.E[start_idx:stop_idx]
    RB_E = np.mean(r_e, axis = 0)
    RB_E_err = np.std(r_e, axis = 0)

    d_m = (chain.m_app - chain.dist_mod - chain.RB[:, None] * chain.E) - \
        (chain.M0_int[:, None] + chain.alpha[:, None] * chain.x + chain.beta_int[:, None] * (chain.c_app - chain.E))

    delta_M = np.mean(d_m[start_idx:stop_idx], axis = 0)
    delta_M_err = np.std(d_m[start_idx:stop_idx], axis = 0)

    fig = plt.figure(figsize=figsize)

    if show_kde:
        gs = fig.add_gridspec(5, 2, width_ratios=[100, 12], hspace=0.05, wspace=0.02)
        ax = np.array([fig.add_subplot(gs[i, 0]) for i in range(5)])
        ax_kde = np.array([fig.add_subplot(gs[i, 1]) for i in range(1, 5)])  # no kde for histogram
        ax_kde[0].sharey(ax[1])
        ax_kde[1].sharey(ax[2])
        ax_kde[2].sharey(ax[3])
        ax_kde[3].sharey(ax[4])
    else:
        gs = fig.add_gridspec(5, 1, hspace=0.05)
        ax = np.array([fig.add_subplot(gs[i, 0]) for i in range(5)])

    # Share x axis across main panels
    for a in ax[1:]:
        a.sharex(ax[0])

    ax[0].hist(host_vec, density = True, bins = n_bins_hist,
               color = hist_color, edgecolor = hist_edge_color, linewidth = 0.5)
    ax[0].set_ylabel(f'{xlabel} density')

    plot_points(ax[1], alpha_x, alpha_x_err)
    ax[1].set_ylabel('$\\alpha x$')

    plot_points(ax[2], beta_int_c, beta_int_c_err)
    ax[2].set_ylabel('$\\beta_{\\rm int} c_{\\rm int}$')

    plot_points(ax[3], RB_E, RB_E_err)
    ax[3].set_ylabel('$R_B E$')

    plot_points(ax[4], delta_M, delta_M_err)
    ax[4].set_ylabel('$\\Delta M_{\\rm int}$')

    if show_kde: # shifted indices (1 less)
        add_kde_marginal(ax_kde[0], alpha_x, alpha_x_err)
        add_kde_marginal(ax_kde[1], beta_int_c, beta_int_c_err)
        add_kde_marginal(ax_kde[2], RB_E, RB_E_err)
        add_kde_marginal(ax_kde[3], delta_M, delta_M_err)

    ax[4].set_xlabel(xlabel)

    if color_vec is not None and color_vec_split_value is not None:
        clbl = clabel if clabel is not None else ''
        legend_elements = [
            Patch(facecolor = pop1_color, label = clbl + f'$\\leq {color_vec_split_value}$'),
            Patch(facecolor = pop2_color, label = clbl + f'$> {color_vec_split_value}$'),
        ]
        ax[0].legend(handles = legend_elements, fontsize = legend_fontsize, framealpha = 0.7)

    x0, x1 = ax[4].get_xlim()
    if x_min is not None:
        x0 = x_min
    if x_max is not None:
        x1 = x_max

    ax[0].set_xlim(x0, x1)

    ax[4].axhline(y = 0, linestyle = 'dashed',
                  color = hline_color, zorder = 10, lw = 1)
    if mass_step_labels_loc is not None:
        ax[4].axhline(y = 0.05, linestyle = 'dashed',
                    color = hline_color, zorder = 10, lw = 1)
        ax[4].axhline(y = -0.05, linestyle = 'dashed',
                    color = hline_color, zorder = 10, lw = 1)
        if mass_step_labels_loc not in ['right', 'left']:
            raise ValueError('mass_step_labels_loc must be either "left", "right", or None')
        ax[4].text(x0 + 0.4 if mass_step_labels_loc == 'left' else x1 - 0.05,
                   0.055, '0.05', color = hline_color,
                   fontsize = 10, ha = 'right', va = 'bottom')
        ax[4].text(x0 + 0.4 if mass_step_labels_loc == 'left' else x1 - 0.05,
                   -0.065, '-0.05', color = hline_color,
                   fontsize = 10, ha = 'right', va = 'top')

    if show_kde:
        ax_kde[3].axhline(y = 0, linestyle = 'dashed',
                          color = hline_color, zorder = 10, lw = 1)
        if mass_step_labels_loc is not None:
            ax_kde[3].axhline(y = 0.05, linestyle = 'dashed',
                        color = hline_color, zorder = 10, lw = 1)
            ax_kde[3].axhline(y = -0.05, linestyle = 'dashed',
                        color = hline_color, zorder = 10, lw = 1)

    if extra_hlines is not None:
        for panel_idx, (val, pos) in extra_hlines.items():
            ax[panel_idx].axhline(y = val, linestyle = 'dashed',
                                 color = hline_color, zorder = 10, lw = 1)
            if show_kde:
                kde_idx = panel_idx - 1
                if 0 <= kde_idx < len(ax_kde):
                    ax_kde[kde_idx].axhline(y=val, linestyle='dashed', color=hline_color,
                                            zorder=10, lw=1)
            if pos == 'left':
                xt = x0 + 0.3
            elif pos == 'right':
                xt = x1 - 0.1
            else:
                continue
            ax[panel_idx].text(xt, val, f'{val}', color = hline_color,
                               fontsize = 10, ha = 'right', va = 'bottom')

    if xval is not None:
        for a in ax:
            a.axvline(x = xval, linestyle = 'dashed',
                    color = vline_color, zorder = 10, lw = 1)

    return fig


def extinguished_magnitude_color_distribution_mean(
    chain: GibbsChainData,
    start_idx: int = 0,
    stop_idx: int = None,
    title: str = None,
    color_dust: bool = True,
    verbose: bool = False,
    labels_fontsize=12,
    title_fontsize=14,
    ticks_fontsize=11,
    text_fontsize=11,
    elinewidth=0.85,
    alpha=0.45,
    capsize=0,
    figsize=(6, 5)
):
    """
    Plot the mean (+-1 std) across MCMC iterations of the extinguished SN population:
    apparent color (c_app) vs. stretch-corrected extinguished magnitude,
    including the R_B slope line with shading for R_B uncertainty.
    """
    params = chain[start_idx:stop_idx]

    c_app = params['c_app']                         # (n_iter, n_SN)
    M_ext = params['m_app'] - params['dist_mod']        # (n_iter, n_SN)
    M_ext_ax = M_ext - params['alpha'][:, np.newaxis] * params['x']
    RB = params['RB']                               # (n_iter,)

    # Mean and std along axis=0 (over iterations), shape (n_SN,)
    c_mean = c_app.mean(axis=0)
    c_std  = c_app.std(axis=0)
    M_mean = M_ext_ax.mean(axis=0)
    M_std  = M_ext_ax.std(axis=0)
    E_mean = params['E'].mean(axis=0)

    rB_mean = RB.mean()
    rB_std  = RB.std()

    # Plot bounds
    c_min, c_max = c_mean.min(), c_mean.max()
    M_min, M_max = M_mean.min(), M_mean.max()
    pad_c = 0.02 * (c_max - c_min) if c_max != c_min else 0.01
    pad_M = 0.02 * (M_max - M_min) if M_max != M_min else 0.1

    fig, ax = plt.subplots(figsize=figsize)

    if color_dust:
        norm = plt.Normalize(vmin=E_mean.min(), vmax=E_mean.max())
        cmap = plt.cm.inferno
        scat = ax.scatter(c_mean, M_mean, s=10, alpha=0.6,
                          c=E_mean, cmap=cmap, norm=norm)
        ax.errorbar(
            c_mean, M_mean,
            xerr=c_std, yerr=M_std,
            fmt='none', ecolor='gray', elinewidth=elinewidth, alpha=alpha, zorder=0,
            capsize=capsize
        )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(
            label='$E$ (dust reddening)' if verbose else '$E$',
            fontsize=labels_fontsize
        )
        cbar.ax.tick_params(labelsize=ticks_fontsize)
    else:
        scat = ax.scatter(c_mean, M_mean, s=10, alpha=0.6, color='k')
        ax.errorbar(
            c_mean, M_mean,
            xerr=c_std, yerr=M_std,
            fmt='none', ecolor='gray', elinewidth=elinewidth, alpha=alpha, zorder=0,
            capsize=capsize
        )

    # Slope line centered at medians of mean arrays
    xvals = np.linspace(c_min - pad_c, c_max + pad_c, 200)
    m0 = np.median(M_mean)
    c0 = np.median(c_mean)

    y_center = m0 + rB_mean * (xvals - c0)
    y_upper  = m0 + (rB_mean + rB_std) * (xvals - c0)
    y_lower  = m0 + (rB_mean - rB_std) * (xvals - c0)

    ax.plot(xvals, y_center, lw=2, color="C0")
    ax.fill_between(xvals, y_lower, y_upper, color="C0", alpha=0.25,
                    label=f"$R_B \\pm 1\\sigma$")

    ax.set_xlim(c_min - pad_c, c_max + pad_c)
    ax.set_ylim(M_max + pad_M, M_min - pad_M)  # reversed y-axis

    ax.set_xlabel(
        "Posterior mean apparent color $\\langle c_{\\rm app} \\rangle$" if verbose else '$\\langle c_{\\rm app} \\rangle$',
        fontsize=labels_fontsize
    )
    ax.set_ylabel(
        "Posterior mean $\\langle M_{\\rm ext} - \\alpha x \\rangle$" if verbose else '$\\langle M_{\\rm ext} - \\alpha x \\rangle$',
        fontsize=labels_fontsize
    )

    if title is None and verbose:
        ax.set_title("Extinguished population (posterior mean $\\pm$ std)", fontsize=title_fontsize)
    else:
        ax.set_title(title, fontsize=title_fontsize)

    ax.text(
        0.65, 0.98, # these values should depend on the fontsize...
        f"$R_B$ = {rB_mean:.2f} $\\pm$ {rB_std:.2f}",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=text_fontsize,
        color="C0"
    )

    ax.tick_params(axis='both', labelsize=ticks_fontsize)
    plt.tight_layout()
    return fig, ax