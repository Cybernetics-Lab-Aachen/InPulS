""" This file defines the iLQG-based trajectory optimization method. """

import copy
import logging

import numpy as np
from numpy.linalg import LinAlgError
import scipy as sp

from gps.algorithm.config import ALG
from gps.algorithm.traj_opt.config import TRAJ_OPT_LQR
from gps.algorithm.algorithm_utils import IterationData, TrajectoryInfo
from gps.utility.general_utils import extract_condition
from gps.algorithm.traj_opt.traj_opt_utils import traj_distr_kl, DGD_MAX_ITER

LOGGER = logging.getLogger(__name__)


class AlgorithmTrajOpt(object):
    """ Sample-based trajectory optimization. """
    def __init__(self, hyperparams):
        config = copy.deepcopy(ALG)
        config.update(hyperparams)
        self._hyperparams = config

        traj_opt_config = copy.deepcopy(TRAJ_OPT_LQR)
        traj_opt_config.update(hyperparams['traj_opt'])
        self.traj_opt_hyperparams = traj_opt_config

        if 'train_conditions' in hyperparams:
            self._cond_idx = hyperparams['train_conditions']
            self.M = len(self._cond_idx)
        else:
            self.M = hyperparams['conditions']
            self._cond_idx = range(self.M)
            self._hyperparams['train_conditions'] = self._cond_idx
            self._hyperparams['test_conditions'] = self._cond_idx
        self.iteration_count = 0

        # Grab a few values from the agent.
        agent = self._hyperparams['agent']
        self.T = self._hyperparams['T'] = agent.T
        self.dU = self._hyperparams['dU'] = agent.dU
        self.dX = self._hyperparams['dX'] = agent.dX
        self.dO = self._hyperparams['dO'] = agent.dO

        init_traj_distr = config['init_traj_distr']
        init_traj_distr['x0'] = agent.x0
        init_traj_distr['dX'] = agent.dX
        init_traj_distr['dU'] = agent.dU
        del self._hyperparams['agent']  # Don't want to pickle this.

        # IterationData objects for each condition.
        self.cur = [IterationData() for _ in range(self.M)]
        self.prev = [IterationData() for _ in range(self.M)]

        dynamics = self._hyperparams['dynamics']
        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()
            self.cur[m].traj_info.dynamics = dynamics['type'](dynamics)
            init_traj_distr = extract_condition(
                self._hyperparams['init_traj_distr'], self._cond_idx[m]
            )
            self.cur[m].traj_distr = init_traj_distr['type'](init_traj_distr)

        self.cost = [
            hyperparams['cost']['type'](hyperparams['cost'])
            for _ in range(self.M)
        ]
        self.base_kl_step = self._hyperparams['kl_step']

    def iteration(self, sample_lists):
        """
        Run iteration of LQR.
        Args:
            sample_lists: List of SampleList objects for each condition.
        """
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]

        # Update dynamics model using all samples.
        self._update_dynamics()

        # Evaluate cost function for all conditions and samples.
        for m in range(self.M):
            self._eval_cost(m)

        # Adjust step size relative to the previous iteration.
        for m in range(self.M):
            if self.iteration_count >= 1 and self.prev[m].sample_list:
                self._stepadjust(m)

        self._update_trajectories()

        self._advance_iteration_variables()

    def _update_dynamics(self):
        """
        Instantiate dynamics objects and update prior. Fit dynamics to
        current samples.
        """
        for m in range(self.M):
            cur_data = self.cur[m].sample_list
            X = cur_data.get_X()
            U = cur_data.get_U()

            # Update prior and fit dynamics.
            self.cur[m].traj_info.dynamics.update_prior(cur_data)
            self.cur[m].traj_info.dynamics.fit(X, U)

            # Fit x0mu/x0sigma.
            x0 = X[:, 0, :]
            x0mu = np.mean(x0, axis=0)
            self.cur[m].traj_info.x0mu = x0mu
            self.cur[m].traj_info.x0sigma = np.diag(
                np.maximum(np.var(x0, axis=0),
                           self._hyperparams['initial_state_var'])
            )

            prior = self.cur[m].traj_info.dynamics.get_prior()
            if prior:
                mu0, Phi, priorm, n0 = prior.initial_state()
                N = len(cur_data)
                self.cur[m].traj_info.x0sigma += \
                        Phi + (N*priorm) / (N+priorm) * \
                        np.outer(x0mu-mu0, x0mu-mu0) / (N+n0)

    def _eval_cost(self, cond):
        """
        Evaluate costs for all samples for a condition.
        Args:
            cond: Condition to evaluate cost on.
        """
        # Constants.
        T, dX, dU = self.T, self.dX, self.dU
        N = len(self.cur[cond].sample_list)

        # Compute cost.
        cs = np.zeros((N, T))
        cc = np.zeros((N, T))
        cv = np.zeros((N, T, dX+dU))
        Cm = np.zeros((N, T, dX+dU, dX+dU))
        for n in range(N):
            sample = self.cur[cond].sample_list[n]
            # Get costs.
            l, lx, lu, lxx, luu, lux = self.cost[cond].eval(sample)
            cc[n, :] = l
            cs[n, :] = l

            # Assemble matrix and vector.
            cv[n, :, :] = np.c_[lx, lu]
            Cm[n, :, :, :] = np.concatenate(
                (np.c_[lxx, np.transpose(lux, [0, 2, 1])], np.c_[lux, luu]),
                axis=1
            )

            # Adjust for expanding cost around a sample.
            X = sample.get_X()
            U = sample.get_U()
            yhat = np.c_[X, U]
            rdiff = -yhat
            rdiff_expand = np.expand_dims(rdiff, axis=2)
            cv_update = np.sum(Cm[n, :, :, :] * rdiff_expand, axis=1)
            cc[n, :] += np.sum(rdiff * cv[n, :, :], axis=1) + 0.5 * \
                    np.sum(rdiff * cv_update, axis=1)
            cv[n, :, :] += cv_update

        # Fill in cost estimate.
        self.cur[cond].traj_info.cc = np.mean(cc, 0)  # Constant term (scalar).
        self.cur[cond].traj_info.cv = np.mean(cv, 0)  # Linear term (vector).
        self.cur[cond].traj_info.Cm = np.mean(Cm, 0)  # Quadratic term (matrix).

        self.cur[cond].cs = cs  # True value of cost.

    def _stepadjust(self, m):
        """
        Calculate new step sizes.
        Args:
            m: Condition
        """
        # Compute values under Laplace approximation. This is the policy
        # that the previous samples were actually drawn from under the
        # dynamics that were estimated from the previous samples.
        previous_laplace_obj = self.estimate_cost(
            self.prev[m].traj_distr, self.prev[m].traj_info
        )
        # This is the policy that we just used under the dynamics that
        # were estimated from the previous samples (so this is the cost
        # we thought we would have).
        new_predicted_laplace_obj = self.estimate_cost(
            self.cur[m].traj_distr, self.prev[m].traj_info
        )

        # This is the actual cost we have under the current trajectory
        # based on the latest samples.
        new_actual_laplace_obj = self.estimate_cost(
            self.cur[m].traj_distr, self.cur[m].traj_info
        )

        # Measure the entropy of the current trajectory (for printout).
        ent = self._measure_ent(m)

        # Compute actual objective values based on the samples.
        previous_mc_obj = np.mean(np.sum(self.prev[m].cs, axis=1), axis=0)
        new_mc_obj = np.mean(np.sum(self.cur[m].cs, axis=1), axis=0)

        LOGGER.debug('Trajectory step: ent: %f cost: %f -> %f',
                     ent, previous_mc_obj, new_mc_obj)

        # Compute predicted and actual improvement.
        predicted_impr = np.sum(previous_laplace_obj) - \
                np.sum(new_predicted_laplace_obj)
        actual_impr = np.sum(previous_laplace_obj) - \
                np.sum(new_actual_laplace_obj)

        # Print improvement details.
        LOGGER.debug('Previous cost: Laplace: %f MC: %f',
                     np.sum(previous_laplace_obj), previous_mc_obj)
        LOGGER.debug('Predicted new cost: Laplace: %f MC: %f',
                     np.sum(new_predicted_laplace_obj), new_mc_obj)
        LOGGER.debug('Actual new cost: Laplace: %f MC: %f',
                     np.sum(new_actual_laplace_obj), new_mc_obj)
        LOGGER.debug('Predicted/actual improvement: %f / %f',
                     predicted_impr, actual_impr)

        self._set_new_mult(predicted_impr, actual_impr, m)

    def estimate_cost(self, traj_distr, traj_info):
        """ Compute Laplace approximation to expected cost. """
        # Constants.
        T = traj_distr.T

        # Perform forward pass (note that we repeat this here, because
        # traj_info may have different dynamics from the ones that were
        # used to compute the distribution already saved in traj).
        mu, sigma = self.forward(traj_distr, traj_info)

        # Compute cost.
        predicted_cost = np.zeros(T)
        for t in range(T):
            predicted_cost[t] = traj_info.cc[t] + 0.5 * \
                    np.sum(sigma[t, :, :] * traj_info.Cm[t, :, :]) + 0.5 * \
                    mu[t, :].T.dot(traj_info.Cm[t, :, :]).dot(mu[t, :]) + \
                    mu[t, :].T.dot(traj_info.cv[t, :])
        return predicted_cost

    def _set_new_mult(self, predicted_impr, actual_impr, m):
        """
        Adjust step size multiplier according to the predicted versus
        actual improvement.
        """
        # Model improvement as I = predicted_dI * KL + penalty * KL^2,
        # where predicted_dI = pred/KL and penalty = (act-pred)/(KL^2).
        # Optimize I w.r.t. KL: 0 = predicted_dI + 2 * penalty * KL =>
        # KL' = (-predicted_dI)/(2*penalty) = (pred/2*(pred-act)) * KL.
        # Therefore, the new multiplier is given by pred/2*(pred-act).
        new_mult = predicted_impr / (2.0 * max(1e-4,
                                               predicted_impr - actual_impr))
        new_mult = max(0.1, min(5.0, new_mult))
        new_step = max(
            min(new_mult * self.cur[m].step_mult,
                self._hyperparams['max_step_mult']),
            self._hyperparams['min_step_mult']
        )
        self.cur[m].step_mult = new_step

        if new_mult > 1:
            LOGGER.debug('Increasing step size multiplier to %f', new_step)
        else:
            LOGGER.debug('Decreasing step size multiplier to %f', new_step)

    def _update_trajectories(self):
        """
        Compute new linear Gaussian controllers.
        """
        if not hasattr(self, 'new_traj_distr'):
            self.new_traj_distr = [
                self.cur[cond].traj_distr for cond in range(self.M)
            ]
        for cond in range(self.M):
            self.new_traj_distr[cond], self.cur[cond].eta = \
                    self.traj_opt_update(cond)

    def traj_opt_update(self, m):
        """ Run dual gradient decent to optimize trajectories. """
        T = self.T
        eta = self.cur[m].eta
        step_mult = self.cur[m].step_mult
        traj_info = self.cur[m].traj_info

        prev_traj_distr = self.cur[m].traj_distr

        # Set KL-divergence step size (epsilon).
        kl_step = T * self.base_kl_step * step_mult

        # We assume at min_eta, kl_div > kl_step, opposite for max_eta.
        min_eta = self.traj_opt_hyperparams['min_eta']
        max_eta = self.traj_opt_hyperparams['max_eta']

        LOGGER.debug("Running DGD for trajectory %d, eta: %f", m, eta)
        for itr in range(DGD_MAX_ITER):
            LOGGER.debug("Iteration %i, bracket: (%.2e , %.2e , %.2e)",
                    itr, min_eta, eta, max_eta)

            # Run fwd/bwd pass, note that eta may be updated.
            # NOTE: we can just ignore case when the new eta is larger.
            traj_distr, eta = self.backward(prev_traj_distr, traj_info,
                                                eta, m)
            new_mu, new_sigma = self.forward(traj_distr, traj_info)

            # Compute KL divergence constraint violation.
            kl_div = traj_distr_kl(new_mu, new_sigma,
                                   traj_distr, prev_traj_distr)
            con = kl_div - kl_step

            # Convergence check - constraint satisfaction.
            if (abs(con) < 0.1*kl_step):
                LOGGER.debug("KL: %f / %f, converged iteration %i",
                        kl_div, kl_step, itr)
                break

            # Choose new eta (bisect bracket or multiply by constant)
            if con < 0: # Eta was too big.
                max_eta = eta
                geom = np.sqrt(min_eta*max_eta)  # Geometric mean.
                new_eta = max(geom, 0.1*max_eta)
                LOGGER.debug("KL: %f / %f, eta too big, new eta: %f",
                        kl_div, kl_step, new_eta)
            else: # Eta was too small.
                min_eta = eta
                geom = np.sqrt(min_eta*max_eta)  # Geometric mean.
                new_eta = min(geom, 10.0*min_eta)
                LOGGER.debug("KL: %f / %f, eta too small, new eta: %f",
                        kl_div, kl_step, new_eta)

            # Logarithmic mean: log_mean(x,y) = (y - x)/(log(y) - log(x))
            eta = new_eta

        if kl_div > kl_step and abs(kl_div - kl_step) > 0.1*kl_step:
            LOGGER.warning(
                "Final KL divergence after DGD convergence is too high."
            )

        return traj_distr, eta

    def backward(self, prev_traj_distr, traj_info, eta, m):
        """
        Perform LQR backward pass. This computes a new linear Gaussian
        policy object.
        Args:
            prev_traj_distr: A linear Gaussian policy object from
                previous iteration.
            traj_info: A TrajectoryInfo object.
            eta: Dual variable.
            m: Condition number.
        Returns:
            traj_distr: A new linear Gaussian policy.
            new_eta: The updated dual variable. Updates happen if the
                Q-function is not PD.
        """
        # Constants.
        T = prev_traj_distr.T
        dU = prev_traj_distr.dU
        dX = prev_traj_distr.dX

        traj_distr = prev_traj_distr.nans_like()

        idx_x = slice(dX)
        idx_u = slice(dX, dX+dU)

        # Pull out dynamics.
        Fm = traj_info.dynamics.Fm
        fv = traj_info.dynamics.fv

        # Non-SPD correction terms.
        del_ = self.traj_opt_hyperparams['del0']
        eta0 = eta

        # Run dynamic programming.
        fail = True
        while fail:
            fail = False  # Flip to true on non-symmetric PD.

            # Allocate.
            Vxx = np.zeros((T, dX, dX))
            Vx = np.zeros((T, dX))

            fCm, fcv = self.compute_costs(m, eta)

            # Compute state-action-state function at each time step.
            for t in range(T - 1, -1, -1):
                # Add in the cost.
                Qtt = fCm[t, :, :]  # (X+U) x (X+U)
                Qt = fcv[t, :]  # (X+U) x 1

                # Add in the value function from the next time step.
                if t < T - 1:
                    multiplier = 1.0
                    Qtt = Qtt + multiplier * \
                            Fm[t, :, :].T.dot(Vxx[t+1, :, :]).dot(Fm[t, :, :])
                    Qt = Qt + multiplier * \
                            Fm[t, :, :].T.dot(Vx[t+1, :] +
                                            Vxx[t+1, :, :].dot(fv[t, :]))

                # Symmetrize quadratic component.
                Qtt = 0.5 * (Qtt + Qtt.T)

                # Compute Cholesky decomposition of Q function action
                # component.
                try:
                    U = sp.linalg.cholesky(Qtt[idx_u, idx_u])
                    L = U.T
                except LinAlgError as e:
                    # Error thrown when Qtt[idx_u, idx_u] is not
                    # symmetric positive definite.
                    LOGGER.debug('LinAlgError: %s', e)
                    fail = True
                    break

                # Store conditional covariance, inverse, and Cholesky.
                traj_distr.inv_pol_covar[t, :, :] = Qtt[idx_u, idx_u]
                traj_distr.pol_covar[t, :, :] = sp.linalg.solve_triangular(
                    U, sp.linalg.solve_triangular(L, np.eye(dU), lower=True)
                )
                traj_distr.chol_pol_covar[t, :, :] = sp.linalg.cholesky(
                    traj_distr.pol_covar[t, :, :]
                )

                # Compute mean terms.
                traj_distr.k[t, :] = -sp.linalg.solve_triangular(
                    U, sp.linalg.solve_triangular(L, Qt[idx_u], lower=True)
                )
                traj_distr.K[t, :, :] = -sp.linalg.solve_triangular(
                    U, sp.linalg.solve_triangular(L, Qtt[idx_u, idx_x],
                                                  lower=True)
                )

                # Compute value function.
                Vxx[t, :, :] = Qtt[idx_x, idx_x] + \
                        Qtt[idx_x, idx_u].dot(traj_distr.K[t, :, :])
                Vx[t, :] = Qt[idx_x] + Qtt[idx_x, idx_u].dot(traj_distr.k[t, :])
                Vxx[t, :, :] = 0.5 * (Vxx[t, :, :] + Vxx[t, :, :].T)

            # Increment eta on non-SPD Q-function.
            if fail:
                old_eta = eta
                eta = eta0 + del_
                LOGGER.debug('Increasing eta: %f -> %f', old_eta, eta)
                del_ *= 2  # Increase del_ exponentially on failure.
                if eta >= 1e16:
                    if np.any(np.isnan(Fm)) or np.any(np.isnan(fv)):
                        raise ValueError('NaNs encountered in dynamics!')
                    raise ValueError('Failed to find PD solution even for very \
                            large eta (check that dynamics and cost are \
                            reasonably well conditioned)!')
        return traj_distr, eta

    def compute_costs(self, m, eta):
        """ Compute cost estimates used in the LQR backward pass. """
        traj_info, traj_distr = self.cur[m].traj_info, self.cur[m].traj_distr
        fCm, fcv = traj_info.Cm / eta, traj_info.cv / eta
        K, ipc, k = traj_distr.K, traj_distr.inv_pol_covar, traj_distr.k

        # Add in the trajectory divergence term.
        for t in range(self.T - 1, -1, -1):
            fCm[t, :, :] += np.vstack([
                np.hstack([
                    K[t, :, :].T.dot(ipc[t, :, :]).dot(K[t, :, :]),
                    -K[t, :, :].T.dot(ipc[t, :, :])
                ]),
                np.hstack([
                    -ipc[t, :, :].dot(K[t, :, :]), ipc[t, :, :]
                ])
            ])
            fcv[t, :] += np.hstack([
                K[t, :, :].T.dot(ipc[t, :, :]).dot(k[t, :]),
                -ipc[t, :, :].dot(k[t, :])
            ])

        return fCm, fcv

    def forward(self, traj_distr, traj_info):
        """
        Perform LQR forward pass. Computes state-action marginals from
        dynamics and policy.
        Args:
            traj_distr: A linear Gaussian policy object.
            traj_info: A TrajectoryInfo object.
        Returns:
            mu: A T x dX mean action vector.
            sigma: A T x dX x dX covariance matrix.
        """
        # Compute state-action marginals from specified conditional
        # parameters and current traj_info.
        T = traj_distr.T
        dU = traj_distr.dU
        dX = traj_distr.dX

        # Constants.
        idx_x = slice(dX)

        # Allocate space.
        sigma = np.zeros((T, dX+dU, dX+dU))
        mu = np.zeros((T, dX+dU))

        # Pull out dynamics.
        Fm = traj_info.dynamics.Fm
        fv = traj_info.dynamics.fv
        dyn_covar = traj_info.dynamics.dyn_covar

        # Set initial covariance (initial mu is always zero).
        sigma[0, idx_x, idx_x] = traj_info.x0sigma
        mu[0, idx_x] = traj_info.x0mu

        for t in range(T):
            sigma[t, :, :] = np.vstack([
                np.hstack([
                    sigma[t, idx_x, idx_x],
                    sigma[t, idx_x, idx_x].dot(traj_distr.K[t, :, :].T)
                ]),
                np.hstack([
                    traj_distr.K[t, :, :].dot(sigma[t, idx_x, idx_x]),
                    traj_distr.K[t, :, :].dot(sigma[t, idx_x, idx_x]).dot(
                        traj_distr.K[t, :, :].T
                    ) + traj_distr.pol_covar[t, :, :]
                ])
            ])
            mu[t, :] = np.hstack([
                mu[t, idx_x],
                traj_distr.K[t, :, :].dot(mu[t, idx_x]) + traj_distr.k[t, :]
            ])
            if t < T - 1:
                sigma[t+1, idx_x, idx_x] = \
                        Fm[t, :, :].dot(sigma[t, :, :]).dot(Fm[t, :, :].T) + \
                        dyn_covar[t, :, :]
                mu[t+1, idx_x] = Fm[t, :, :].dot(mu[t, :]) + fv[t, :]
        return mu, sigma

    def _advance_iteration_variables(self):
        """
        Move all 'cur' variables to 'prev', and advance iteration
        counter.
        """
        self.iteration_count += 1
        self.prev = copy.deepcopy(self.cur)
        # TODO: change IterationData to reflect new stuff better
        for m in range(self.M):
            self.prev[m].new_traj_distr = self.new_traj_distr[m]
        self.cur = [IterationData() for _ in range(self.M)]
        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()
            self.cur[m].traj_info.dynamics = copy.deepcopy(self.prev[m].traj_info.dynamics)
            self.cur[m].step_mult = self.prev[m].step_mult
            self.cur[m].eta = self.prev[m].eta
            self.cur[m].traj_distr = self.new_traj_distr[m]
        delattr(self, 'new_traj_distr')

    def _measure_ent(self, m):
        """ Measure the entropy of the current trajectory. """
        ent = 0
        for t in range(self.T):
            ent = ent + np.sum(
                np.log(np.diag(self.cur[m].traj_distr.chol_pol_covar[t, :, :]))
            )
        return ent
