"""This file defines the linear Gaussian policy class."""
import numpy as np

from gps.algorithm.policy.policy import Policy
from gps.utility.general_utils import check_shape


class LinearGaussianPolicy(Policy):
    """Time-varying linear Gaussian policy.

    U = K*x + k + noise, where noise ~ N(0, chol_pol_covar)
    """

    def __init__(self, K, k, pol_covar, chol_pol_covar, inv_pol_covar):
        Policy.__init__(self)

        # Assume K has the correct shape, and make sure others match.
        self.T = K.shape[0]
        self.dU = K.shape[1]
        self.dX = K.shape[2]

        check_shape(k, (self.T, self.dU))
        check_shape(pol_covar, (self.T, self.dU, self.dU))
        check_shape(chol_pol_covar, (self.T, self.dU, self.dU))
        check_shape(inv_pol_covar, (self.T, self.dU, self.dU))

        self.K = K
        self.k = k
        self.pol_covar = pol_covar
        self.chol_pol_covar = chol_pol_covar
        self.inv_pol_covar = inv_pol_covar

    def act(self, x, obs, t, noise=None):
        """Decides an action for the given state/observation at the current timestep.

        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: A dU-dimensional noise vector.

        Returns:
            A dU dimensional action vector.

        """
        u = self.K[t].dot(x) + self.k[t]
        if noise is not None:
            covar = self.chol_pol_covar[t].T

            u += covar.dot(noise[t])
        return u

    def nans_like(self):
        """Creates a new linear Gaussian policy object with the same dimensions but all values filled with NaNs."""
        policy = LinearGaussianPolicy(
            np.zeros_like(self.K),
            np.zeros_like(self.k),
            np.zeros_like(self.pol_covar),
            np.zeros_like(self.chol_pol_covar),
            np.zeros_like(self.inv_pol_covar),
        )
        policy.K.fill(np.nan)
        policy.k.fill(np.nan)
        policy.pol_covar.fill(np.nan)
        policy.chol_pol_covar.fill(np.nan)
        policy.inv_pol_covar.fill(np.nan)
        return policy
