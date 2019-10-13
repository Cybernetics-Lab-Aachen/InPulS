"""This file defines the base cost class."""
from abc import ABC, abstractmethod


class Cost(ABC):
    """Abstract cost superclass."""

    def __init__(self, hyperparams):
        """Initializes the cost function.

        Args:
            hyperparams: Dictionary of hyperparameters.

        """
        self._hyperparams = hyperparams

    @abstractmethod
    def eval(self, sample):
        """Evaluates the function and it's derivatives.

        Args:
            sample:  A single sample.

        """
        pass
