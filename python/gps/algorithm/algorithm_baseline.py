import copy
import logging

import numpy as np

from gps.algorithm.algorithm import Algorithm, Timer
from gps.algorithm.algorithm_utils import IterationData, TrajectoryInfo

LOGGER = logging.getLogger(__name__)


class AlgorithmBaseline(Algorithm):
    def __init__(self, hyperparams):
        Algorithm.__init__(self, hyperparams)

        self.policy_opt = self._hyperparams['policy_opt']['type'](self._hyperparams['policy_opt'], self.dO, self.dU)

    def iteration(self, sample_lists, itr):
        # Store the samples and evaluate the costs.
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]
            self._eval_cost(m)

        with Timer(self.timers, 'pol_update'):
            self._update_policy()

        # Prepare for next iteration
        self._advance_iteration_variables()

    def _update_policy(self, initial_policy=False):
        dU, dO, T = self.dU, self.dO, self.T
        N = len(self.cur[0].sample_list)

        # Gather states, actions and immediate costs for all samples
        X = np.empty((self.M, N, T, dO))
        U = np.empty((self.M, N, T, dU))
        cs = np.empty((self.M, N, T))

        # Iterate over conditions m
        for m in range(self.M):
            samples = self.cur[m].sample_list
            cs[m] = self.cur[m].cs

            # Iterate over samples n
            # Get time-indexed actions.
            for n in range(N):
                X[m, n] = samples[n].get_X()
                U[m, n] = samples[n].get_U()

        self.policy_opt.update(X=X, U=U, cs=cs, initial_policy=initial_policy)

    def _advance_iteration_variables(self):
        """
        Move all 'cur' variables to 'prev', and advance iteration
        counter.
        """
        self.iteration_count += 1
        self.prev = copy.deepcopy(self.cur)
        self.cur = [IterationData() for _ in range(self.M)]
        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()
