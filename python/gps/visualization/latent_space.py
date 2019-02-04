import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


def visualize_latent_space(
    file_name, z_mean, z_std, x_label='$\\mathbf{z}$', y_label='pdf', show=False, export_data=True
):
    """
    Visualizes approximation ability.
    Args:
        file_name: File name without extension.
        Z: ndarray (N, dZ) with latent states.
        show: Display generated plot. This is a blocking operation.
        export_data: Writes a npz file containing the plotted data points.
                     This is useful for later recreation of the plot.
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.grid(linestyle=':')

    N, dZ = z_mean.shape

    x = np.linspace(-3, 3, 1000).reshape(1000, 1)
    plt.plot(x, sp.stats.norm.pdf(x), color="black", linestyle=":", label='$\\mathcal{N}(0,1)$')
    for dim in range(dZ):
        # Fit GMM by hand
        gmm = GaussianMixture(N)
        gmm.means_ = z_mean[:, dim].reshape(N, 1)
        gmm.precisions_cholesky_ = (1 / z_std[:, dim]).reshape(N, 1, 1)
        gmm.weights_ = np.ones(N) / N
        ax1.plot(x, np.exp(gmm.score_samples(x)), linewidth=1, label='$\\mathbf{z}[%d]$' % dim)

    ax1.legend()
    fig.savefig(file_name + ".png", bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close(fig)
    if export_data:
        np.savez_compressed(file_name, z_mean=z_mean, z_std=z_std)
