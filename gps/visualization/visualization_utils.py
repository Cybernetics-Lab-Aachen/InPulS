import numpy as np


def aggregate(data, axis=0, mode='auto'):
    N = data.shape[axis]

    if '=' in mode:
        split = mode.split('=', 1)
        mode = split[0]
        mode_option = split[1]
    else:
        mode_option = None

    if mode == 'span' or mode == 'auto' and N < 5:
        data_mean = np.mean(data, axis=axis)
        data_min = np.amin(data, axis=axis)
        data_max = np.amax(data, axis=axis)
    elif mode == 'quartiles' or mode == 'auto' and N < 25:
        data_mean = np.quantile(data, 0.5, axis=axis)
        data_min = np.quantile(data, 0.25, axis=axis)
        data_max = np.quantile(data, 0.75, axis=axis)
    elif mode == 'std':
        data_mean = np.mean(data, axis=axis)
        std = np.std(data, axis=axis)
        stds = int(mode_option) if mode_option is not None else 1  # Use e.g. std=3 to specify three standard deviations
        data_min = data_mean - std * stds
        data_max = data_mean + std * stds
    else:
        raise ValueError('Unknown mode "%s"' % mode)

    return data_mean, data_min, data_max
