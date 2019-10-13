"""This file defines torque (action) costs."""
import copy

import numpy as np

from gps.algorithm.cost import Cost
from gps.algorithm.cost.config import COST_ACTION


class CostAction(Cost):
    """Costs applied to the actions."""

    def __init__(self, hyperparams):
        """Initializes the cost function.

        Args:
            hyperparams: Dictionary of hyperparameters.

        """
        config = copy.deepcopy(COST_ACTION)
        config.update(hyperparams)
        Cost.__init__(self, config)

    def eval(self, sample):
        """Evaluates cost function and derivatives on a sample.

        Args:
            sample: A single sample

        """
        sample_u = sample.get_U() - self._hyperparams['target_state']
        T = sample.T
        Du = sample.dU
        Dx = sample.dX
        l = 0.5 * np.sum(self._hyperparams['wu'] * (sample_u**2), axis=1)
        lu = self._hyperparams['wu'] * sample_u
        lx = np.zeros((T, Dx))
        luu = np.tile(np.diag(self._hyperparams['wu']), [T, 1, 1])
        lxx = np.zeros((T, Dx, Dx))
        lux = np.zeros((T, Du, Dx))
        return l, lx, lu, lxx, luu, lux
