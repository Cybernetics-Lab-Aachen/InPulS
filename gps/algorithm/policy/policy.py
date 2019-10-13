"""This file defines the base class for policies."""
from abc import ABC, abstractmethod


class Policy(ABC):
    """Computes actions from states/observations."""

    @abstractmethod
    def act(self, x, obs, t, noise):
        """Decides an action for the given state/observation at the current timestep.

        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: A dU-dimensional noise vector.

        Returns:
            A dU dimensional action vector.

        """
        pass
