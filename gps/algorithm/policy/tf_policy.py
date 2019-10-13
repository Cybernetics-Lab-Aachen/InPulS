import numpy as np
import tensorflow as tf

from gps.algorithm.policy.policy import Policy


class TfPolicy(Policy):
    """A neural network policy implemented in tensor flow.

    The network output is taken to be the mean, and Gaussian noise is added on top of it.

    U = net.forward(obs) + noise, where noise ~ N(0, diag(var))

    Args:
        obs_tensor: tensor representing tf observation. Used in feed dict for forward pass.
        act_op: tf op to execute the forward pass. Use sess.run on this op.
        var: Du-dimensional noise variance vector.
        sess: tf session.
        device_string: tf device string for running on either gpu or cpu.

    """

    def __init__(self, dU, obs_tensor, act_op, var, sess, device_string):
        Policy.__init__(self)
        self.dU = dU
        self.obs_tensor = obs_tensor
        self.act_op = act_op
        self.sess = sess
        self.device_string = device_string
        self.chol_pol_covar = np.diag(np.sqrt(var))
        self.scale = None  # must be set from elsewhere based on observations
        self.bias = None
        self.x_idx = None

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
        # Normalize obs.
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        obs[:, self.x_idx] = obs[:, self.x_idx].dot(self.scale) + self.bias
        with tf.device(self.device_string):
            action_mean = self.sess.run(self.act_op, feed_dict={self.obs_tensor: obs})[0]
        if noise is None:
            u = action_mean
        else:
            covar = self.chol_pol_covar.T
            if t is None:
                u = action_mean + covar.dot(noise[0])
            else:
                u = action_mean + covar.dot(noise[t])
        return u  # the DAG computations are batched by default, but we use batch size 1.
