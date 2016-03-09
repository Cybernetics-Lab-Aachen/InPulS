import pickle

import numpy as np
import tensorflow as tf

from gps.algorithm.policy.policy import Policy


class TfPolicy(Policy):
    """
    A neural network policy implemented in tensor flow. The network output is
    taken to be the mean, and Gaussian noise is added on top of it.
    U = net.forward(obs) + noise, where noise ~ N(0, diag(var))
    Args:
        test_net: Initialized tf network that can run forward.
        var: Du-dimensional noise variance vector.
    """
    def __init__(self, obs_tensor, act_op, var, sess, device_string):
        Policy.__init__(self)
        self.chol_pol_covar = np.diag(np.sqrt(var))
        self.obs_tensor = obs_tensor
        self.act_op = act_op
        self.sess = sess
        self.device_string = device_string  # is it even worth running this on the gpu? Won't comm time dominate?
        self.scale = None  # must be set from elsewhere based on observations
        self.bias = None

    def act(self, x, obs, t, noise):
        """
        Return an action for a state.
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: Action noise. This will be scaled by the variance.
        """
        # Normalize obs.
        obs = obs.dot(self.scale) + self.bias
        with tf.device(self.device_string):
            action_mean = self.sess.run(self.act_op, feed_dict={self.obs_tensor: np.expand_dims(obs, 0)})
        if noise is None:
            u = action_mean
        else:
            u = action_mean + self.chol_pol_covar.T.dot(noise)
        return u

    def pickle_policy(self, deg_obs, deg_action, checkpoint_path):
        """
        We can save just the policy if we are only interested in running forward.
        """
        pickled_pol = {'deg_obs': deg_obs, 'deg_action': deg_action, 'chol_pol_covar': self.chol_pol_covar,
                       'checkpoint_path_tf': checkpoint_path + '_tf_data', 'scale': self.scale, 'bias': self.bias,
                       'device_string': self.device_string}
        pickle.dump(pickled_pol, open(checkpoint_path, "wb"))
        saver = tf.train.Saver()
        saver.save(self.sess, checkpoint_path + '_tf_data')

    @classmethod
    def load_policy(cls, policy_dict_path, tf_generator):
        """
        For when we only need the forward pass. For instance, to run on the robot from
        a checkpointed policy.
        """
        from tensorflow.python.framework import ops
        ops.reset_default_graph()  # we need to destroy the default graph before re_init or checkpoint won't restore.
        pol_dict = pickle.load(open(policy_dict_path, "rb"))
        tf_map = tf_generator(dim_input=pol_dict['deg_obs'], dim_output=pol_dict['deg_action'], batch_size=None)

        sess = tf.Session()
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        saver = tf.train.Saver()
        check_file = pol_dict['checkpoint_path_tf']
        saver.restore(sess, check_file)

        device_string = pol_dict['device_string']

        cls_init = cls(tf_map.get_input_tensor(), tf_map.get_act_op(), np.zeros((1,)), sess, device_string)
        cls_init.chol_pol_covar = pol_dict['chol_pol_covar']
        return cls_init

