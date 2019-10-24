"""This file defines policy optimization for a tensorflow policy."""
import copy
import logging

import numpy as np

# NOTE: Order of these imports matters for some reason.
# Changing it can lead to segmentation faults on some machines.

from gps.algorithm.policy_opt.config import POLICY_OPT_TF
import tensorflow as tf

from gps.algorithm.policy.tf_policy import TfPolicy
from gps.algorithm.policy_opt.policy_opt import PolicyOpt
from gps.algorithm.policy_opt.tf_utils import TfSolver

LOGGER = logging.getLogger(__name__)


class PolicyOptTf(PolicyOpt):
    """Policy optimization using tensor flow for DAG computations/nonlinear function approximation."""

    def __init__(self, hyperparams, dO, dU):
        config = copy.deepcopy(POLICY_OPT_TF)
        config.update(hyperparams)

        PolicyOpt.__init__(self, config, dO, dU)

        tf.set_random_seed(self._hyperparams['random_seed'])

        self.tf_iter = 0
        self.checkpoint_file = self._hyperparams['checkpoint_prefix']
        save_path = self._hyperparams['save_path']
        self.model_file = (save_path + "model.ckpt")
        self.batch_size = self._hyperparams['batch_size']
        self.device_string = "/cpu:0"
        if self._hyperparams['use_gpu'] == 1:
            self.gpu_device = self._hyperparams['gpu_id']
            self.device_string = "/gpu:" + str(self.gpu_device)
        self.act_op = None  # mu_hat
        self.loss_scalar = None
        self.obs_tensor = None
        self.precision_tensor = None
        self.action_tensor = None  # mu true
        self.solver = None
        self.init_network()
        self.init_solver()
        self.var = self._hyperparams['init_var'] * np.ones(dU)
        self.sess = tf.Session()
        self.policy = TfPolicy(dU, self.obs_tensor, self.act_op, np.zeros(dU), self.sess, self.device_string)
        # List of indices for state (vector) data and image (tensor) data in observation.
        self.x_idx, self.img_idx, i = [], [], 0
        if 'obs_image_data' not in self._hyperparams['network_params']:
            self._hyperparams['network_params'].update({'obs_image_data': []})
        for sensor in self._hyperparams['network_params']['obs_include']:
            dim = self._hyperparams['network_params']['sensor_dims'][sensor]
            if sensor in self._hyperparams['network_params']['obs_image_data']:
                self.img_idx = self.img_idx + list(range(i, i + dim))
            else:
                self.x_idx = self.x_idx + list(range(i, i + dim))
            i += dim
        init_op = tf.initialize_all_variables()
        self.saver = tf.train.Saver(max_to_keep=None)
        self.sess.run(init_op)

    def init_network(self):
        """Helper method to initialize the tf networks used."""
        tf_map_generator = self._hyperparams['network_model']
        tf_map = tf_map_generator(
            dim_input=self._dO,
            dim_output=self._dU,
            batch_size=self.batch_size,
            network_config=self._hyperparams['network_params']
        )
        self.obs_tensor = tf_map.get_input_tensor()
        self.action_tensor = tf_map.get_target_output_tensor()
        self.precision_tensor = tf_map.get_precision_tensor()
        self.act_op = tf_map.get_output_op()
        self.loss_scalar = tf_map.get_loss_op()

    def init_solver(self):
        """Helper method to initialize the solver."""
        self.solver = TfSolver(
            loss_scalar=self.loss_scalar,
            solver_name=self._hyperparams['solver_type'],
            base_lr=self._hyperparams['lr'],
            lr_policy=self._hyperparams['lr_policy'],
            momentum=self._hyperparams['momentum'],
            weight_decay=self._hyperparams['weight_decay']
        )

    def update(self, X, mu, prc, **kwargs):
        """Update policy.

        Args:
            obs: Numpy array of observations, N x T x dO.
            tgt_mu: Numpy array of mean controller outputs, N x T x dU.
            tgt_prc: Numpy array of precision matrices, N x T x dU x dU.

        Returns:
            A tensorflow object with updated weights.

        """
        M, N, T = X.shape[:3]
        N_train = M * N * T
        dU, dO = self._dU, self._dO

        # TODO - Make sure all weights are nonzero?

        # Reshape inputs.
        X = np.reshape(X, (N_train, dO))
        mu = np.reshape(mu, (N_train, dU))
        prc = np.reshape(np.repeat(prc[:, None], N, axis=1), (N_train, dU, dU))
        prc.setflags(write=False)

        # TODO: Find entries with very low weights?

        # Normalize X, but only compute normalzation at the beginning.
        if self.policy.scale is None or self.policy.bias is None:
            self.policy.x_idx = self.x_idx
            # 1e-3 to avoid infs if some state dimensions don't change in the
            # first batch of samples
            self.policy.scale = np.diag(1.0 / np.maximum(np.std(X[:, self.x_idx], axis=0), 1e-3))
            self.policy.bias = -np.mean(X[:, self.x_idx].dot(self.policy.scale), axis=0)
        X[:, self.x_idx] = X[:, self.x_idx].dot(self.policy.scale) + self.policy.bias

        # Assuming that N*T >= self.batch_size.
        batches_per_epoch = int(N_train / self.batch_size)
        assert batches_per_epoch * self.batch_size == N_train, \
            '%d * %d != %d' % (batches_per_epoch, self.batch_size, N_train)
        idx = np.arange(N_train)
        average_loss = 0
        np.random.shuffle(idx)

        # actual training.
        for i in range(self._hyperparams['iterations']):
            # Load in data for this batch.
            start_idx = int(i * self.batch_size % (batches_per_epoch * self.batch_size))
            idx_i = idx[start_idx:start_idx + self.batch_size]
            feed_dict = {self.obs_tensor: X[idx_i], self.action_tensor: mu[idx_i], self.precision_tensor: prc[idx_i]}
            train_loss = self.solver(feed_dict, self.sess)

            average_loss += train_loss
            if (i + 1) % 500 == 0:
                LOGGER.debug('tensorflow iteration %d, average loss %f', i + 1, average_loss / 500)
                print('supervised tf loss is ' + str(average_loss))
                average_loss = 0

        # Keep track of tensorflow iterations for loading solver states.
        self.tf_iter += self._hyperparams['iterations']

        # Optimize variance.
        A = (np.sum(prc, 0) + 2 * N * T * self._hyperparams['ent_reg'] * np.ones((dU, dU))) / N_train

        # TODO - Use dense covariance?
        self.var = 1 / np.diag(A)
        self.policy.chol_pol_covar = np.diag(np.sqrt(self.var))

        return self.policy

    def prob(self, obs):
        """Run policy forward.

        Args:
            obs: Numpy array of observations that is N x T x dO.

        """
        dU = self._dU
        N, T = obs.shape[:2]

        # Normalize obs.
        if self.policy.scale is not None:
            # TODO: Should prob be called before update?
            for n in range(N):
                obs[n, :, self.x_idx] = (obs[n, :, self.x_idx].T.dot(self.policy.scale) + self.policy.bias).T

        output = np.zeros((N, T, dU))

        for i in range(N):
            for t in range(T):
                # Feed in data.
                feed_dict = {self.obs_tensor: np.expand_dims(obs[i, t], axis=0)}
                with tf.device(self.device_string):
                    output[i, t, :] = self.sess.run(self.act_op, feed_dict=feed_dict)

        pol_sigma = np.tile(np.diag(self.var), [N, T, 1, 1])
        pol_prec = np.tile(np.diag(1.0 / self.var), [N, T, 1, 1])
        pol_det_sigma = np.tile(np.prod(self.var), [N, T])

        return output, pol_sigma, pol_prec, pol_det_sigma

    def restore_model(self, data_files_dir, iteration_count):
        """Loads the network weighs from a file."""
        self._data_files_dir = data_files_dir
        self.iteration_count = iteration_count
        self.saver.restore(self.sess, self._data_files_dir + 'model-%02d' % (self.iteration_count))

    def store_model(self):
        """Saves the network weighs in a file."""
        self.saver.save(self.sess, self._data_files_dir + 'model-%02d' % (self.iteration_count))
