"""This file defines the sample list wrapper and sample writers."""
import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


class SampleList:
    """Class that handles writes and reads to sample data."""

    def __init__(self, samples):
        """Initializes the sample list.

        Args:
            samples: Array of samples.

        """
        self._samples = samples

    def get(self, sensor_name, idx=None):
        """Returns N x T x dX numpy array of states."""
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get(sensor_name) for i in idx])

    def get_X(self, idx=None):
        """Returns N x T x dX numpy array of states."""
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get_X() for i in idx])

    def get_U(self, idx=None):
        """Returns N x T x dU numpy array of actions."""
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get_U() for i in idx])

    def get_obs(self, idx=None):
        """Returns N x T x dO numpy array of features."""
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get_obs() for i in idx])

    def get_samples(self, idx=None):
        """Returns N sample objects."""
        if idx is None:
            idx = range(len(self._samples))
        return [self._samples[i] for i in idx]

    def __len__(self):
        """Returns number of samples."""
        return len(self._samples)

    def __getitem__(self, idx):
        """Returns sample by index."""
        return self.get_samples([idx])[0]
