import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import Normalizer


def visualize_linear_model(
    file_name,
    coeff,
    intercept,
    cov,
    x,
    y=None,
    N=100,
    coeff_label='Coefficients',
    intercept_label='Intercept',
    cov_label='Covariance',
    y_label='$\\mathbf{y}$',
    time_label='$t$',
    show=False,
    export_data=True
):
    """
    Creates a figure visualizing a timeseries of linear Gausian models.

    Args:
        file: File to write figure to.
        coeff: Linear coefficients. Shape: (T, dY, dX)
        intercept: Constants. Shape: (T, dY)
        cov: Covariances. Shape: (T, dY, dY)
        x: Shape (T, dX)
        y: Optional. Shape (T, dY)
        N: Number of random samples drawn to visualize variance.
    """
    fig = plt.figure(figsize=(16, 12))

    T, dX = x.shape
    _, dY = intercept.shape

    # Check shapes
    assert coeff.shape == (T, dY, dX)
    assert intercept.shape == (T, dY)
    assert cov.shape == (T, dY, dY)
    assert x.shape == (T, dX)
    if y is not None:
        assert y.shape == (T, dY)

    # Intercept
    ax1 = fig.add_subplot(221)
    ax1.set_ylabel(intercept_label)
    ax1.set_xlabel(time_label)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.grid(linestyle=':')
    for dim in range(dY):
        line, = ax1.plot(np.arange(T), intercept[:, dim], linewidth=1)

    # Coefficients
    ax2 = fig.add_subplot(222)
    ax2.set_ylabel(coeff_label)
    ax2.set_xlabel(time_label)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.grid(linestyle=':')
    for dim1 in range(dY):
        for dim2 in range(dX):
            line, = ax2.plot(np.arange(T, dtype=int), coeff[:, dim1, dim2], linewidth=1)

    # Covariance
    ax3 = fig.add_subplot(223)
    ax3.set_ylabel(cov_label)
    ax3.set_xlabel(time_label)
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.grid(linestyle=':')
    for dim1 in range(dY):
        for dim2 in range(dY):
            line, = ax3.plot(np.arange(T), cov[:, dim1, dim2], linewidth=1)

    # Approximation
    y_ = np.empty((N, T, dY))  # Approx y using the model
    for t in range(T):
        mu = np.dot(coeff[t], x[t]) + intercept[t]
        y_[:, t] = np.random.multivariate_normal(mean=mu, cov=cov[t], size=N)
    y_mean = np.mean(y_, axis=0)
    y_std = np.std(y_, axis=0)

    ax4 = fig.add_subplot(224)
    ax4.set_ylabel(y_label)
    ax4.set_xlabel(time_label)
    ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax4.grid(linestyle=':')
    for dim in range(dY):
        line, = ax4.plot(np.arange(T), y_mean[:, dim], linewidth=1)
        c = line.get_color()
        if y is not None:
            ax4.plot(np.arange(T), y[:, dim], ':', color=c)
        ax4.fill_between(
            np.arange(T),
            y_mean[:, dim] - y_std[:, dim],
            y_mean[:, dim] + y_std[:, dim],
            facecolor=c,
            alpha=0.25,
            interpolate=True
        )

    if file_name is not None:
        fig.savefig(file_name + ".pdf", bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close(fig)
    if export_data:
        np.savez_compressed(file_name, coeff=coeff, intercept=intercept, cov=cov, y=y, y_mean=y_mean, y_std=y_std)


def visualize_K(file_name, K, labels=None, show=False):
    """
    Visualizes training losses.
    Args:
        file_name: File name without extension.
        losses: ndarray (N_epochs, N_losses) with losses.
        labels: list (N_losses, ) with labels for each loss.
    """
    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('%')
    ax1.grid(linestyle=':')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    T, dX = K.shape

    K = np.abs(Normalizer(norm='l1').fit_transform(K))

    if labels is None:
        labels = ['$\\mathbf{x}_{%d}$' % dim for dim in range(dX)]

    ax1.stackplot(
        np.arange(T), K.T, labels=labels, colors=plt.cm.nipy_spectral((1 + np.arange(dX, dtype=float)) / (dX + 1))
    )

    ax1.legend()
    ax1.set_xlim(left=0, right=T - 1)
    ax1.set_ylim(bottom=0, top=1)

    if file_name is not None:
        fig.savefig(file_name + ".pdf", bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close(fig)
