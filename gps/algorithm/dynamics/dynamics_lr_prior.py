"""This file defines linear regression with an arbitrary prior."""
import numpy as np

from gps.algorithm.dynamics import Dynamics


class DynamicsLRPrior(Dynamics):
    """Linear dynamics fitted via linear regression with arbitrary prior."""

    def __init__(self, hyperparams):
        """Initializes the dynamics.

        Args:
            hyperparams: Dictionary of hyperparameters.

        """
        Dynamics.__init__(self, hyperparams)
        self.Fm = None
        self.fv = None
        self.dyn_covar = None
        self.prior = self._hyperparams['prior']['type'](self._hyperparams['prior'])

    def update_prior(self, samples):
        """Update dynamics prior."""
        X = samples.get_X()
        U = samples.get_U()
        self.prior.update(X, U)

    def get_prior(self):
        """Returns prior object."""
        return self.prior

    def fit(self, X, U, prior_only=False):
        """Fit dynamics."""
        # Constants
        N, T, dimX = X.shape
        dimU = U.shape[2]

        index_xu = slice(dimX + dimU)
        index_x = slice(dimX + dimU, dimX + dimU + dimX)

        sig_reg = np.zeros((dimX + dimU + dimX, dimX + dimU + dimX))
        sig_reg[index_xu, index_xu] = self._hyperparams['regularization']

        # Weights used in computing sample mean and sample covariance.
        dwts = (1.0 / N) * np.ones(N)
        D = np.diag((1.0 / (N - 1)) * np.ones(N))

        # Allocate
        self.Fm = np.zeros([T, dimX, dimX + dimU])
        self.fv = np.zeros([T, dimX])
        self.dyn_covar = np.zeros([T, dimX, dimX])

        for t in range(T - 1):
            Ys = np.c_[X[:, t, :], U[:, t, :], X[:, t + 1, :]]

            # Obtain normal-inverse-Wishart prior.
            mu0, Phi, mm, n0 = self.prior.eval(dimX, dimU, Ys)

            # Compute empirical mean and covariance.
            empmu = np.sum((Ys.T * dwts).T, axis=0)
            diff = Ys - empmu
            empsig = diff.T.dot(D).dot(diff)
            # Symmetrize empsig to counter numerical errors.
            empsig = 0.5 * (empsig + empsig.T)

            # Compute posterior estimates of mean and covariance.
            mu = empmu if not prior_only else mu0
            sigma = (Phi + (N - 1) * empsig + (N * mm) /
                     (N + mm) * np.outer(empmu - mu0, empmu - mu0)) / (N + n0) if not prior_only else Phi
            # Symmetrize sigma to counter numerical errors.
            sigma = 0.5 * (sigma + sigma.T)
            # Add sigma regularization.
            sigma += sig_reg

            # Conditioning to get dynamics.
            Fm = np.linalg.solve(sigma[index_xu, index_xu], sigma[index_xu, index_x]).T
            fv = mu[index_x] - Fm.dot(mu[index_xu])
            dyn_covar = sigma[index_x, index_x] - Fm.dot(sigma[index_xu, index_xu]).dot(Fm.T)
            # Symmetrize dyn_covar to counter numerical errors.
            dyn_covar = 0.5 * (dyn_covar + dyn_covar.T)

            self.Fm[t, :, :] = Fm
            self.fv[t, :] = fv
            self.dyn_covar[t, :, :] = dyn_covar

        return self.Fm, self.fv, self.dyn_covar
