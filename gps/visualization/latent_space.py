"""Tools for visualizing the latent state of an autoencoder."""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


def export_latent_space_data(file_name, x_train, z_mean_train, z_std_train, x_test, z_mean_test, z_std_test):
    """Export samples from a latent space."""
    np.savez_compressed(
        file_name,
        x_train=x_train,
        z_mean_train=z_mean_train,
        z_std_train=z_std_train,
        x_test=x_test,
        z_mean_test=z_mean_test,
        z_std_test=z_std_test
    )


def visualize_latent_space(file_name, z_mean, z_std, x_label='$\\mathbf{z}$', y_label='pdf', show=False):
    """Visualizes approximation ability.

    Args:
        file_name: File name without extension.
        z_mean: ndarray (N, dZ) with latent states.
        z_std: ndarray (N, dZ, dZ) with std of latent states.
        x_label: ALbel of the x-axis.
        y_label: ALbel of the y-axis.
        show: Display generated plot. This is a blocking operation.

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
        fig.savefig(file_name + ".pdf", bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close(fig)


def visualize_latent_space_tsne(
    file_name,
    x_train,
    z_train,
    x_test,
    z_test,
    color_map='gist_rainbow',
    N_colors=16,
    perplexity_scale=5,
    show=False,
    export_data=True
):
    """Visualizes latent space via tsne.

    Args:
        file_name: File name without extension.
        x_train: ndarray (N, dX) of states.
        z_train: ndarray (N, S, dZ) with latent states were N is the number of input states and S the number of latent
            space sample of each state.
        show: Display generated plot. This is a blocking operation.
        export_data: Writes a npz file containing the plotted data points. This is useful for later recreation of the
            plot.

    """
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.grid(linestyle=':')

    kmeans = KMeans(n_clusters=N_colors).fit(x_train)
    N_train, S, dZ = z_train.shape
    N_test, _, _ = z_test.shape

    z = np.concatenate([z_train, z_test])
    z_embedded = TSNE(
        n_components=2, perplexity=S * perplexity_scale, init='pca', n_iter=1000
    ).fit_transform(z.reshape(-1, dZ))

    # Plot trained
    plt.scatter(
        x=z_embedded[:N_train * S, 0],
        y=z_embedded[:N_train * S, 1],
        # x=z_train[:,0, 0],
        # y=z_train[:, 0,1],
        c=np.repeat(kmeans.labels_, S),
        cmap=plt.cm.get_cmap(color_map, N_colors),
        edgecolor='',
        marker="o",
        alpha=0.5,
    )

    # Plot test
    nearest = sp.spatial.cKDTree(x_train).query(x_test)[0]
    plt.scatter(
        x=z_embedded[N_train * S:, 0],
        y=z_embedded[N_train * S:, 1],
        # x=z_test[:,0, 0],
        # y=z_test[:, 0,1],
        c=np.repeat(1 - nearest / np.amax(nearest), S),
        cmap=plt.cm.gray,
        edgecolor='black',
        linewidth=0.5,
        marker=".",
    )
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    if file_name is not None:
        fig.savefig(file_name + ".pdf", bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close(fig)
    if export_data:
        np.savez_compressed(file_name, x_train=x_train, z_train=z_train, x_test=x_test, z_test=z_test)
