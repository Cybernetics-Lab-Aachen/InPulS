"""This file defines the base trajectory optimization class."""
from abc import ABC, abstractmethod


class TrajOpt(ABC):
    def __init__(self, hyperparams):
        self._hyperparams = hyperparams

    @abstractmethod
    def update(self):
        """Update trajectory distributions."""
        pass
