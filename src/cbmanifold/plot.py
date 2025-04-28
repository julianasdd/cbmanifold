## Decomposition functions

import numpy as np
import matplotlib.pyplot as plt


def plot_linear_model(lm, t_range=(-250, 251), axs=None):
    tt = np.arange(*t_range)
    cells = np.arange(1, lm.dim + 1)

    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(6, 5), sharey=True)
        print(f"figsize: {fig.get_size_inches()}")

    p1 = axs[0].pcolormesh(
        tt,
        cells,
        lm.rate,
        cmap="seismic",
        vmin=-np.abs(lm.rate).max(),
        vmax=np.abs(lm.rate).max(),
    )
    p2 = axs[1].pcolormesh(
        tt,
        cells,
        lm.drate,
        cmap="seismic",
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
        fig, ax = plt.subplots(figsize=(1.25, 3))

    dims = np.arange(1, dmax + 1)
    vars_to_plot = var_explained[:dmax]
    ax.plot(dims, vars_to_plot, ":k", linewidth=0.5)
    ax.plot(dims, vars_to_plot, "or")
    ax.set_ylabel("variance explained (%)")
    ax.set_xlabel("dimension")
    ax.set_xlim(-0.5, dmax + 1)
    ax.set_xticks(np.arange(1, dmax + 1, 9))
    plt.tight_layout()

    return (fig, ax)
