""" This file defines linear regression with an arbitrary prior. """
import numpy as np

from gps.algorithm.dynamics.dynamics import Dynamics


class DynamicsLRPrior(Dynamics):
    """ Dynamics with linear regression, with arbitrary prior. """
    def __init__(self, hyperparams):
        Dynamics.__init__(self, hyperparams)
        self.Fm = None
        self.fv = None
        self.dyn_covar = None
        self.prior = \
                self._hyperparams['prior']['type'](self._hyperparams['prior'])

    def update_prior(self, samples):
        """ Update dynamics prior. """
        X = samples.get_X()
        U = samples.get_U()
        self.prior.update(X, U)

    def get_prior(self):
        """ Return the dynamics prior. """
        return self.prior

    def fit(self, X, U):
        """ Fit dynamics. """
        N, T, dimX = X.shape
        dimU = U.shape[2]

        if N == 1:
            raise ValueError("Cannot fit dynamics on 1 sample")

        # Allocate 
        self.Fm = np.zeros([T, dimX, dimX + dimU])
        self.fv = np.zeros([T, dimX])
        self.dyn_covar = np.zeros([T, dimX, dimX])

        index_XU = slice(dimX + dimU)

        sig_reg = np.zeros((dimX + dimU + dimX, dimX + dimU + dimX))
        sig_reg[index_XU, index_XU] = self._hyperparams['regularization']

        # Fit dynamics with least squares regression.
        dwts = (1.0 / N) * np.ones(N)
        for t in range(T - 1):
            Ys = np.c_[X[:, t, :], U[:, t, :], X[:, t + 1, :]]

            # Obtain Normal-inverse-Wishart prior.
            mu0, Phi, mm, n0 = self.prior.eval(dimX, dimU, Ys)

            # Obtain posterior and condition on [X, U]
            Fm, fv, dyn_covar = self.gauss_fit_joint_prior(Ys,
                        mu0, Phi, mm, n0, dwts, dimX, dimU, sig_reg)

            self.Fm[t, :, :] = Fm
            self.fv[t, :] = fv
            self.dyn_covar[t, :, :] = dyn_covar
        return self.Fm, self.fv, self.dyn_covar

    def gauss_fit_joint_prior(self, Ys, mu0, Phi, m, n0, dwts, dimX, dimU, sig_reg):
        """ Perform Gaussian fit to data with a prior. """
        # Build weights matrix.
        D = np.diag(dwts)
        # Compute empirical mean and covariance.
        mun = np.sum((Ys.T * dwts).T, axis=0)
        diff = Ys - mun
        empsig = diff.T.dot(D).dot(diff)
        # Symmetrize empsig to counter numerical errors.
        empsig = 0.5 * (empsig + empsig.T)

        # MAP estimate of joint distribution.
        N = dwts.shape[0]
        mu = mun
        sigma = (N * empsig + Phi + (N * m) / (N + m) *
                 np.outer(mun - mu0, mun - mu0)) / (N + n0)
        # Symmetrize sigma to counter numerical errors.
        sigma = 0.5 * (sigma + sigma.T)
        # Add sigma regularization.
        sigma += sig_reg

        # Conditioning to get dynamics.
        index_XU = slice(dimX + dimU)
        index_X = slice(dimX + dimU, dimX + dimU + dimX)
        Fm = np.linalg.solve(sigma[index_XU, index_XU],
                             sigma[index_XU, index_X]).T
        fv = mu[index_X] - Fm.dot(mu[index_XU])
        dyn_covar = sigma[index_X, index_X] - Fm.dot(sigma[index_XU, index_XU]).dot(Fm.T)
        # Symmetrize dyn_covar to counter numerical errors.
        dyn_covar = 0.5 * (dyn_covar + dyn_covar.T)

        return Fm, fv, dyn_covar
