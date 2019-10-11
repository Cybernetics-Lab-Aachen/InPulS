import numpy as np
import matplotlib.pyplot as plt

from gps.algorithm.cost.cost_sum import CostSum


def __plot_costs(ax, samples, cf, weight=1.0):
    if isinstance(cf, CostSum):
        for i, sub_cf in enumerate(cf._costs):
            __plot_costs(ax, samples, sub_cf, weight * cf._weights[i])
    else:
        N = len(samples)
        T = samples[0].T

        costs = np.empty((N, T))
        for n in range(N):
            costs[n] = cf.eval(samples[n])[0] * weight

        costs_mean = np.mean(costs, axis=0)
        costs_min = np.amin(costs, axis=0)
        costs_max = np.amax(costs, axis=0)
        line, = ax.plot(np.arange(T), costs_mean, label=cf._hyperparams['name'])
        c = line.get_color()
        ax.fill_between(np.arange(T), costs_min, costs_max, facecolor=c, alpha=0.25, interpolate=True)


def visualize_costs(file_name, samples, cost_function, show=False):
    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('cost')

    # Plot costs
    __plot_costs(ax1, samples, cost_function)

    ax1.grid(linestyle=":")

    ax1.legend(fontsize='x-small')

    if file_name is not None:
        fig.savefig(file_name + ".pdf", bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close(fig)
