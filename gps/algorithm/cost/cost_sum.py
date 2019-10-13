"""This file defines a cost sum of arbitrary other costs."""
import copy

from gps.algorithm.cost import Cost
from gps.algorithm.cost.config import COST_SUM


class CostSum(Cost):
    """A wrapper cost function that adds other cost functions."""

    def __init__(self, hyperparams):
        """Initializes the cost function.

        Args:
            hyperparams: Dictionary of hyperparameters.

        """
        config = copy.deepcopy(COST_SUM)
        config.update(hyperparams)
        Cost.__init__(self, config)

        self._costs = []
        self._weights = self._hyperparams['weights']

        for cost in self._hyperparams['costs']:
            self._costs.append(cost['type'](cost))

    def eval(self, sample):
        """Evaluates cost function and derivatives on a sample.

        Args:
            sample: A single sample

        """
        l, lx, lu, lxx, luu, lux = self._costs[0].eval(sample)

        # Compute weighted sum of each cost value and derivatives.
        weight = self._weights[0]
        l = l * weight
        lx = lx * weight
        lu = lu * weight
        lxx = lxx * weight
        luu = luu * weight
        lux = lux * weight
        for i in range(1, len(self._costs)):
            pl, plx, plu, plxx, pluu, plux = self._costs[i].eval(sample)
            weight = self._weights[i]
            l = l + pl * weight
            lx = lx + plx * weight
            lu = lu + plu * weight
            lxx = lxx + plxx * weight
            luu = luu + pluu * weight
            lux = lux + plux * weight
        return l, lx, lu, lxx, luu, lux
