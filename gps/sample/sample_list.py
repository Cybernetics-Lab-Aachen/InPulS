""" This file defines the sample list wrapper and sample writers. """
import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


class SampleList(object):
    """ Class that handles writes and reads to sample data. """

    def __init__(self, samples):
        self._samples = samples

    def get(self, sensor_name, idx=None):
        """ Returns N x T x dX numpy array of states. """
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get(sensor_name) for i in idx])

    def get_X(self, idx=None):
        """ Returns N x T x dX numpy array of states. """
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get_X() for i in idx])

    def get_EEF_Position(self, idx=None):
        """ Returns N x T x dX numpy array of states. """
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get_EEF_Position() for i in idx])

    def get_U(self, idx=None):
        """ Returns N x T x dU numpy array of actions. """
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get_U() for i in idx])

    def get_obs(self, idx=None):
        """ Returns N x T x dO numpy array of features. """
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get_obs() for i in idx])

    def get_samples(self, idx=None):
        """ Returns N sample objects. """
        if idx is None:
            idx = range(len(self._samples))
        return [self._samples[i] for i in idx]

    def num_samples(self):
        """ Returns number of samples. """
        return len(self._samples)

    # Convenience methods.
    def __len__(self):
        return self.num_samples()

    def __getitem__(self, idx):
        return self.get_samples([idx])[0]