import numpy as np
import matplotlib.pyplot as plt


def visualize_traj_opt(file_name, mu, dX, dU, alpha_discount=0.75, show=False):
    """Visualizes traj_opt progress."""
    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('cost')

    N, T, _ = mu.shape

    traj_opt_U = np.asarray(mu)[:, :, dX:]
    for dim in range(dU):
        line, = ax1.plot(np.arange(T), traj_opt_U[0, :, dim], label='$\\mathbf{u}_{%d}$' % dim)
        c = line.get_color()

        alpha = 1.0
        for n in range(1, traj_opt_U.shape[0]):
            alpha *= alpha_discount
            ax1.plot(np.arange(T), traj_opt_U[n, :, dim], color=c, alpha=alpha)

    ax1.grid(linestyle=":")

    ax1.legend()

    if file_name is not None:
        fig.savefig(file_name + ".pdf", bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close(fig)
