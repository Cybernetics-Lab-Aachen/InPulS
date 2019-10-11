import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from gps.visualization.visualization_utils import aggregate
from os.path import isfile


def eval_samples(experiment, metric, sample_type='samples_pol-random', cond=None):
    """
    Finds pol samples files and evaluate trajectories
    """
    itr = 0
    pol_samples = []

    while True:
        sample_file = experiment + sample_type + '_%02d.npz' % itr
        if not isfile(sample_file):
            break
        pol_samples.append(sample_file)
        itr += 1

    iterations = len(pol_samples)
    assert iterations > 0, experiment

    X = np.load(pol_samples[0])['X']
    M, N, T, dX = X.shape
    if cond is None:
        X = X.reshape(M * N, T, dX)
    else:
        X = X[cond]

    evals = np.empty((iterations, N))
    for i in range(iterations):
        X = np.load(pol_samples[i])['X']
        for n in range(N):
            evals[i, n] = metric(X[0, n])
    return evals


def visualize_training(
    file_name,
    experiments,
    metric,
    target=None,
    mode='auto',
    x_label='iterations',
    y_label='distance',
    show=False,
    figsize=(16, 9),
):
    """
    Visualizes training iterations.
    Args:
        file_name: File name without extension.
        losses: ndarray (N_epochs, N_losses) with losses.
        labels: list (N_losses, ) with labels for each loss.
    """
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.grid(linestyle=':')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    if target is not None:
        ax1.axhline(target, linestyle='--', color='grey', label='target')

    for ex in experiments:
        data = eval_samples(ex['experiment'], metric, sample_type=ex['sample_type'])
        T, _ = data.shape
        xs = (np.arange(T) + 1) * ex.get('N_per_itr', 1)
        eval_mean, eval_min, eval_max = aggregate(data, axis=1, mode=mode)
        line, = ax1.plot(xs, eval_mean, label=ex['label'])
        c = line.get_color()
        ax1.fill_between(xs, eval_min, eval_max, facecolor=c, alpha=0.25, interpolate=True)

    ax1.legend()

    if file_name is not None:
        fig.savefig(file_name + ".pdf", bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close(fig)
