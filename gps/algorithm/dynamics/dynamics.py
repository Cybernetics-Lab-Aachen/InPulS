"""This file defines the base class for dynamics estimation."""
from abc import ABC, abstractmethod

import numpy as np


class Dynamics(ABC):
    """Abstract dynamics superclass."""

    def __init__(self, hyperparams):
        """Initializes the dynamics.

        Args:
            hyperparams: Dictionary of hyperparameters.

        """
        self._hyperparams = hyperparams

        # Fitted dynamics: x_t+1 = Fm * [x_t;u_t] + fv.
        self.Fm = np.array(np.nan)  # Linear component
        self.fv = np.array(np.nan)  # Constant component
        self.dyn_covar = np.array(np.nan)  # Covariance.

    @abstractmethod
    def update_prior(self, sample):
        """Update dynamics prior."""
        pass

    @abstractmethod
    def get_prior(self):
        """Returns prior object."""
        pass

    @abstractmethod
    def fit(self, sample_list):
        """Fit dynamics."""
        pass

    def copy(self):
        """Return a copy of this dynamics object."""
        dyn = type(self)(self._hyperparams)
        dyn.Fm = np.copy(self.Fm)
        dyn.fv = np.copy(self.fv)
        dyn.dyn_covar = np.copy(self.dyn_covar)
        return dyn
