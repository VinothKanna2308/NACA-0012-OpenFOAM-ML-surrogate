"""Plot helpers for the C-mesh: overview, leading edge and trailing edge views."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _draw(ax, grid, xlim, ylim, stride=1, lw=0.4):
    X, Y = grid.X, grid.Y
    for i in range(0, X.shape[0], stride):
        ax.plot(X[i], Y[i], "k-", lw=lw)
    for j in range(0, X.shape[1], stride):
        ax.plot(X[:, j], Y[:, j], "k-", lw=lw)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")


def plot(grid, path, with_negative=True):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    _draw(axs[0], grid, (-3, 4), (-3, 3), stride=2, lw=0.3)
    axs[0].set_title("overview")
    _draw(axs[1], grid, (-0.04, 0.12), (-0.08, 0.08), lw=0.4)
    axs[1].set_title("leading edge")
    _draw(axs[2], grid, (0.9, 1.15), (-0.1, 0.1), lw=0.4)
    axs[2].set_title("trailing edge / wake start")
    if with_negative:
        X, Y = grid.X, grid.Y
        x00, x10, x11, x01 = X[:-1, :-1], X[1:, :-1], X[1:, 1:], X[:-1, 1:]
        y00, y10, y11, y01 = Y[:-1, :-1], Y[1:, :-1], Y[1:, 1:], Y[:-1, 1:]
        area = 0.5 * ((x00 * y10 - x10 * y00) + (x10 * y11 - x11 * y10)
                      + (x11 * y01 - x01 * y11) + (x01 * y00 - x00 * y01))
        bad = np.argwhere(area <= 0)
        if len(bad):
            for ax in axs:
                ax.plot(0.25 * (x00 + x10 + x11 + x01)[area <= 0],
                        0.25 * (y00 + y10 + y11 + y01)[area <= 0], "r.", ms=3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
