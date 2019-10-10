""" This file defines code for iLQG-based trajectory optimization. """
import logging
import copy

import numpy as np
from numpy.linalg import LinAlgError
import scipy as sp

from gps.algorithm.traj_opt.config import TRAJ_OPT_LQR
from gps.algorithm.traj_opt.traj_opt import TrajOpt
from gps.algorithm.traj_opt.traj_opt_utils import calc_traj_distr_kl, DGD_MAX_ITER

from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS

LOGGER = logging.getLogger(__name__)


class TrajOptLQRPython(TrajOpt):
    """ LQR trajectory optimization, Python implementation. """

    def __init__(self, hyperparams):
        config = copy.deepcopy(TRAJ_OPT_LQR)
        config.update(hyperparams)

        TrajOpt.__init__(self, config)

    def update(self, m, algorithm, initial_update=False):
        """ Run dual gradient decent to optimize trajectories. """
        T = algorithm.T
        eta = algorithm.cur[m].eta
        step_mult = algorithm.cur[m].step_mult
        traj_info = algorithm.cur[m].traj_info

        if isinstance(algorithm, AlgorithmMDGPS) and not initial_update:
            # For MDGPS, constrain to previous NN linearization
            prev_traj_distr = algorithm.cur[m].pol_info.traj_distr()
        else:
            # For BADMM/trajopt, constrain to previous LG controller
            prev_traj_distr = algorithm.cur[m].traj_distr

        # Set KL-divergence step size (epsilon).
        kl_step = T * algorithm.base_kl_step * step_mult

        # We assume at min_eta, kl_div > kl_step, opposite for max_eta.
        min_eta = self._hyperparams['min_eta']
        max_eta = self._hyperparams['max_eta']

        LOGGER.debug("Running DGD for trajectory %d, eta: %f", m, eta)
        mus = []
        for itr in range(DGD_MAX_ITER):
            LOGGER.debug("Iteration %i, bracket: (%.2e , %.2e , %.2e)", itr, min_eta, eta, max_eta)

            # Run fwd/bwd pass, note that eta may be updated.
            # NOTE: we can just ignore case when the new eta is larger.
            traj_distr, eta = self.backward(prev_traj_distr, traj_info, eta, algorithm, m)
            new_mu, new_sigma = self.forward(traj_distr, traj_info)
            mus.append(new_mu)

            # Compute KL divergence constraint violation.
            kl_div, _ = calc_traj_distr_kl(new_mu, new_sigma, traj_distr, prev_traj_distr)
            con = kl_div - kl_step

            # Convergence check - constraint satisfaction.
            if (abs(con) < 0.1 * kl_step):
                LOGGER.debug("KL: %f / %f, converged iteration %i", kl_div, kl_step, itr)
                break

            # Choose new eta (bisect bracket or multiply by constant)
            if con < 0:  # Eta was too big.
                max_eta = eta
                geom = np.sqrt(min_eta * max_eta)  # Geometric mean.
                new_eta = max(geom, 0.1 * max_eta)
                LOGGER.debug("KL: %f / %f, eta too big, new eta: %f", kl_div, kl_step, new_eta)
            else:  # Eta was too small.
                min_eta = eta
                geom = np.sqrt(min_eta * max_eta)  # Geometric mean.
                new_eta = min(geom, 10.0 * min_eta)
                LOGGER.debug("KL: %f / %f, eta too small, new eta: %f", kl_div, kl_step, new_eta)

            # Logarithmic mean: log_mean(x,y) = (y - x)/(log(y) - log(x))
            eta = new_eta

        if kl_div > kl_step and abs(kl_div - kl_step) > 0.1 * kl_step:
            LOGGER.warning("Final KL divergence after DGD convergence is too high.")

        from gps.visualization.traj_opt import visualize_traj_opt
        visualize_traj_opt(
            file_name=self._data_files_dir + 'traj_opt_m%d-%02d' % (m, self.iteration_count),
            mu=np.asarray(mus),
            dX=traj_distr.dX,
            dU=traj_distr.dU
        )

        return traj_distr, eta, new_mu, new_sigma

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
            predicted_cost[t] = traj_info.cc[t] + 0.5 * np.sum(
                sigma[t, :, :] * traj_info.Cm[t, :, :]
            ) + 0.5 * mu[t, :].T.dot(traj_info.Cm[t, :, :]).dot(mu[t, :]) + mu[t, :].T.dot(traj_info.cv[t, :])
        return predicted_cost

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
        sigma = np.zeros((T, dX + dU, dX + dU))
        mu = np.zeros((T, dX + dU))

        # Pull out dynamics.
        Fm = traj_info.dynamics.Fm
        fv = traj_info.dynamics.fv
        dyn_covar = traj_info.dynamics.dyn_covar

        # Set initial covariance (initial mu is always zero).
        sigma[0, idx_x, idx_x] = traj_info.x0sigma
        mu[0, idx_x] = traj_info.x0mu

        for t in range(T):
            sigma[t, :, :] = np.vstack(
                [
                    np.hstack([sigma[t, idx_x, idx_x], sigma[t, idx_x, idx_x].dot(traj_distr.K[t, :, :].T)]),
                    np.hstack(
                        [
                            traj_distr.K[t, :, :].dot(sigma[t, idx_x, idx_x]),
                            traj_distr.K[t, :, :].dot(sigma[t, idx_x, idx_x]).dot(traj_distr.K[t, :, :].T) +
                            traj_distr.pol_covar[t, :, :]
                        ]
                    )
                ]
            )
            mu[t, :] = np.hstack([mu[t, idx_x], traj_distr.K[t, :, :].dot(mu[t, idx_x]) + traj_distr.k[t, :]])
            if t < T - 1:
                sigma[t + 1, idx_x, idx_x] = Fm[t, :, :].dot(sigma[t, :, :]).dot(Fm[t, :, :].T) + dyn_covar[t, :, :]
                mu[t + 1, idx_x] = Fm[t, :, :].dot(mu[t, :]) + fv[t, :]

            # Symmetrize sigma
            sigma[t] = (sigma[t] + sigma[t].T) / 2
        return mu, sigma

    def backward(self, prev_traj_distr, traj_info, eta, algorithm, m):
        """
        Perform LQR backward pass. This computes a new linear Gaussian
        policy object.
        Args:
            prev_traj_distr: A linear Gaussian policy object from
                previous iteration.
            traj_info: A TrajectoryInfo object.
            eta: Dual variable.
            algorithm: Algorithm object needed to compute costs.
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
        idx_u = slice(dX, dX + dU)

        # Pull out dynamics.
        Fm = traj_info.dynamics.Fm
        fv = traj_info.dynamics.fv

        # Non-SPD correction terms.
        del_ = self._hyperparams['del0']
        eta0 = eta

        # Run dynamic programming.
        fail = True
        while fail:
            fail = False  # Flip to true on non-symmetric PD.

            # Allocate.
            Vxx = np.zeros((T, dX, dX))
            Vx = np.zeros((T, dX))

            fCm, fcv = algorithm.compute_costs(m, eta)
            self.Cm_ext = fCm
            self.cv_ext = fcv

            #if algorithm.inner_itr == 0:
            #    print("Optimize without importance sampling")
            # Compute state-action-state function at each time step.
            for t in range(T - 1, -1, -1):
                # Add in the cost.
                Qtt = fCm[t, :, :]  # (X+U) x (X+U)
                Qt = fcv[t, :]  # (X+U) x 1

                # Add in the value function from the next time step.
                if t < T - 1:
                    Qtt = Qtt + Fm[t, :, :].T.dot(Vxx[t + 1, :, :]).dot(Fm[t, :, :])
                    Qt = Qt + Fm[t, :, :].T.dot(Vx[t + 1, :] + Vxx[t + 1, :, :].dot(fv[t, :]))

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
                traj_distr.chol_pol_covar[t, :, :] = sp.linalg.cholesky(traj_distr.pol_covar[t, :, :])

                # Compute mean terms.
                traj_distr.k[t] = -sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(L, Qt[idx_u], lower=True))
                traj_distr.K[t, :, :] = -sp.linalg.solve_triangular(
                    U, sp.linalg.solve_triangular(L, Qtt[idx_u, idx_x], lower=True)
                )

                # Compute value function.
                Vxx[t, :, :] = Qtt[idx_x, idx_x] + Qtt[idx_x, idx_u].dot(traj_distr.K[t, :, :])
                Vx[t, :] = Qt[idx_x] + Qtt[idx_x, idx_u].dot(traj_distr.k[t, :])
                Vxx[t, :, :] = 0.5 * (Vxx[t, :, :] + Vxx[t, :, :].T)

                traj_distr.Qm[t, :, :] = Qtt
                traj_distr.qv[t, :] = Qt

            # Increment eta on non-SPD Q-function.
            if fail:
                old_eta = eta
                eta = eta0 + del_
                LOGGER.debug('Increasing eta: %f -> %f', old_eta, eta)
                del_ *= 2  # Increase del_ exponentially on failure.
                if eta >= 1e16:
                    if np.any(np.isnan(Fm)) or np.any(np.isnan(fv)):
                        raise ValueError('NaNs encountered in dynamics!')
                    raise ValueError(
                        'Failed to find PD solution even for very \
                            large eta (check that dynamics and cost are \
                            reasonably well conditioned)!'
                    )
        return traj_distr, eta
