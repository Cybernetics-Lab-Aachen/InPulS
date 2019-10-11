import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle


def visualize_trajectories(file_name, X, U, X_labels=None, U_labels=None, show=False):
    N, T, dX = X.shape
    _, _, dU = U.shape

    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax1.set_xlabel('$t$')
    ax2.set_xlabel('$t$')
    ax1.set_ylabel('$x$')
    ax2.set_ylabel('$u$')

    # Create labels
    if X_labels is None:
        X_labels = ['$x_{%d}$' % dim for dim in range(dX)]
    if U_labels is None:
        U_labels = ['$u_{%d}$' % dim for dim in range(dU)]

    # Plot states
    X_mean = np.mean(X, axis=0)
    X_min = np.amin(X, axis=0)
    X_max = np.amax(X, axis=0)
    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)
    for dim in range(dX):
        line, = ax1.plot(np.arange(T), X_mean[:, dim], next(linecycler), label=X_labels[dim])
        c = line.get_color()
        ax1.fill_between(np.arange(T), X_min[:, dim], X_max[:, dim], facecolor=c, alpha=0.25, interpolate=True)

    # Plot actions
    U_mean = np.mean(U, axis=0)
    U_min = np.amin(U, axis=0)
    U_max = np.amax(U, axis=0)
    for dim in range(dU):
        line, = ax2.plot(np.arange(T), U_mean[:, dim], next(linecycler), label=U_labels[dim])
        c = line.get_color()
        ax2.fill_between(np.arange(T), U_min[:, dim], U_max[:, dim], facecolor=c, alpha=0.25, interpolate=True)

    ax1.grid(linestyle=":")
    ax2.grid(linestyle=":")

    ax1.legend(fontsize='x-small')
    ax2.legend(fontsize='x-small')

    if file_name is not None:
        fig.savefig(file_name + ".pdf", bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close(fig)
