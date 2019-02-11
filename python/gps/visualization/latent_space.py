import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE


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

    xs = np.linspace(-3, 3, 1000).reshape(1000, 1)
    plt.plot(xs, sp.stats.norm.pdf(xs), color="black", linestyle=":", label='$\\mathcal{N}(0,1)$')
    for dim in range(dZ):
        # Fit GMM by hand
        gmm = GaussianMixture(N)
        gmm.means_ = z_mean[:, dim].reshape(N, 1)
        gmm.precisions_cholesky_ = (1 / z_std[:, dim]).reshape(N, 1, 1)
        gmm.weights_ = np.ones(N) / N
        ax1.plot(xs, np.exp(gmm.score_samples(xs)), linewidth=1, label='$\\mathbf{z}[%d]$' % dim)

    ax1.legend()

    if file_name is not None:
        fig.savefig(file_name + ".png", bbox_inches='tight', pad_inches=0)
        if export_data:
            np.savez_compressed(file_name, z_mean=z_mean, z_std=z_std)
    if show:
        plt.show()
    plt.close(fig)

def visualize_latent_space_tsne(file_name, z, color_map='gist_rainbow', show=False, export_data=True):
    """
    Visualizes latent space via tsne.
    Args:
        file_name: File name without extension.
        Z: ndarray (N, S, dZ) with latent states were N is the number of input states and S the number of laten space sample of each state.
        show: Display generated plot. This is a blocking operation.
        export_data: Writes a npz file containing the plotted data points.
                     This is useful for later recreation of the plot.
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.grid(linestyle=':')

    N, S, dZ = z.shape
    z_embedded = TSNE(n_components=2, perplexity=S).fit_transform(z.reshape(-1, dZ))

    plt.scatter(z_embedded[:, 0], z_embedded[:, 1], c=np.repeat(np.arange(N), S), cmap=plt.cm.get_cmap(color_map, N))
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    if file_name is not None:
        fig.savefig(file_name + ".png", bbox_inches='tight', pad_inches=0)
        if export_data:
            np.savez_compressed(file_name, z=z)
    if show:
        plt.show()
    plt.close(fig)
