import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_linear_model(lm, t_range=(-250, 251), cmap="seismic", axs=None):
    tt = np.arange(*t_range)
    cells = np.arange(1, lm.rate.shape[0] + 1)

    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(6, 5), sharey=True)
        print(f"figsize: {fig.get_size_inches()}")

    p1 = axs[0].pcolormesh(
        tt,
        cells,
        lm.rate,
        cmap=cmap,
        vmin=-np.abs(lm.rate).max(),
        vmax=np.abs(lm.rate).max(),
    )
    p2 = axs[1].pcolormesh(
        tt,
        cells,
        lm.drate,
        cmap=cmap,
        vmin=-np.abs(lm.drate).max(),
        vmax=np.abs(lm.drate).max(),
    )

    ## Add colorbars
    cbars = []
    cbars.append(axs[0].figure.colorbar(p1, ax=axs[0], shrink=0.5))
    cbars[-1].set_label("rate (Hz)", size=6)
    cbars.append(axs[1].figure.colorbar(p2, ax=axs[1], shrink=0.5))
    cbars[-1].set_label("∂rate (Hz/unit)", size=6)

    for cbar in cbars:
        cbar.ax.tick_params(labelsize=6)

    for i in range(2):
        axs[i].set_xlabel("time after onset, (ms)")

    axs[0].set_ylabel(lm.label)
    axs[0].set_title("rate")
    axs[1].set_title("∂rate")

    return (fig, axs), cbars


def plot_var_explained(var_explained, dmax=15, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(1.25, 2))

    dims = np.arange(1, dmax + 1)
    vars_to_plot = var_explained[:dmax]
    ax.plot(dims, vars_to_plot, ":k", linewidth=0.5)
    ax.plot(dims, vars_to_plot, "o", markerfacecolor="w", markeredgecolor="k")
    ax.set_ylabel("variance explained (%)")
    ax.set_xlabel("dimension")
    ax.set_xlim(-0.5, dmax + 1)
    ax.set_xticks(np.arange(1, dmax + 1, 9))
    plt.tight_layout()

    return (fig, ax)


# ## plots related to manifolds


def cmap_pv(v0s):
    """Generate colormap for peak velocity plots."""
    n_colors = len(v0s)
    cmap = plt.cm.coolwarm(np.linspace(0, 1, n_colors))
    return cmap


def plot_ld_pv_time(lm, v0s, npanels, cmap_func=None, fig=None):
    """Plot linear dimensions against peak velocity over time.

    Parameters:
    -----------
    lm : LinearModel
        Linear model containing p (projections) and dp (derivatives)
    v0s : array-like
        Array of peak velocities to plot
    npanels : int
        Number of dimensions to plot
    cmap_func : callable, optional
        Function to generate colormap. If None, uses default cmap_pv

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure handle
    axs : list of matplotlib.axes.Axes
        List of axis handles
    """
    if cmap_func is None:
        cmap_func = cmap_pv

    tt = np.arange(-250, 251)
    cmap = cmap_func(v0s)

    # Create figure with custom layout
    if fig is None:
        fig = plt.figure(figsize=(8, 2 * npanels))

    gs = GridSpec(
        npanels,
        1,
        figure=fig,
        hspace=0.1 * 2 / npanels,
        left=0.225,
        right=0.95,
        top=0.785,
        bottom=0.075,
    )

    axs = []
    # Create panels
    for k in range(npanels):
        ax = fig.add_subplot(gs[k])
        for iv, v0 in enumerate(v0s):
            sig = lm.p[k, :] + (v0 - lm.v0) * lm.dp[k, :]
            ax.plot(tt, sig, color=cmap[iv], linewidth=1)

        # Format the axis
        ax.set_xlim(-200, 200)
        ax.set_xticks(np.arange(-200, 201, 100))
        if k < npanels - 1:
            ax.set_xticklabels([])
        ax.set_ylabel(f"Dimension {k+1}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        axs.append(ax)

    # Set xlabel only on bottom panel
    axs[-1].set_xticklabels(np.arange(-200, 201, 100))
    axs[-1].set_xlabel("Time from onset [ms]")

    # Link x axes
    for ax in axs[:-1]:
        ax.get_shared_x_axes().join(ax, axs[-1])

    return fig, axs
