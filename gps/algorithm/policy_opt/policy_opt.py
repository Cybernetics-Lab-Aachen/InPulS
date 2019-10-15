"""This file defines the base policy optimization class."""
from abc import ABC, abstractmethod


class PolicyOpt(ABC):
    def __init__(self, hyperparams, dO, dU):
        self._hyperparams = hyperparams
        self._dO = dO
        self._dU = dU

    @abstractmethod
    def update(self):
        """Update policy."""
        pass
