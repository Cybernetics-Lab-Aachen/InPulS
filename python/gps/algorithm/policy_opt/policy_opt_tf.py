""" This file defines policy optimization for a tensorflow policy. """
import copy
import logging

import numpy as np

import tensorflow as tf

from gps.algorithm.policy.tf_policy import TfPolicy
from gps.algorithm.policy_opt.policy_opt import PolicyOpt
from gps.algorithm.policy_opt.config import POLICY_OPT_TF

LOGGER = logging.getLogger(__name__)


class TfSolver:
    """ A container for holding solver hyperparams in tensorflow. Used to execute backwards pass. """
    def __init__(self, loss_scalar, solver_name='adam', snapshot_prefix=None, base_lr=None, lr_policy=None,
                 momentum=None, weight_decay=None):
        self.snapshot_prefix = snapshot_prefix
        self.base_lr = base_lr
        self.lr_policy = lr_policy
        self.momentum = momentum
        self.solver_name = solver_name
        self.loss_scalar = loss_scalar

        if self.lr_policy is not 'fixed':
            raise NotImplemented('learning rate policies other than fixed are not implemented')

        self.weight_decay = weight_decay
        if weight_decay is not None:
            trainable_vars = tf.trainable_variables()
            loss_with_reg = self.loss_scalar
            for var in trainable_vars:
                loss_with_reg += self.weight_decay*tf.nn.l2_loss(var)
            self.loss_scalar = loss_with_reg

        self.solver_op = self.get_solver_op()

    def get_solver_op(self):
        solver_string = self.solver_name.lower()
        if solver_string is 'adam':
            return tf.train.AdamOptimizer(learning_rate=self.base_lr).minimize(self.loss_scalar)
        elif solver_string is 'rmsprop':
            return tf.train.RMSPropOptimizer(learning_rate=self.base_lr, decay=self.momentum).minimize(self.loss_scalar)
        elif solver_string is 'momentum':
            return tf.train.MomentumOptimizer(learning_rate=self.base_lr, momentum=self.momentum).minimize(self.loss_scalar)
        elif solver_string is 'adagrad':
            return tf.train.AdagradOptimizer(learning_rate=self.base_lr, initial_accumulator_value=self.momentum)
        elif solver_string is 'gd':
            return tf.train.GradientDescentOptimizer(learning_rate=self.base_lr)
        else:
            raise NotImplementedError("Please select a valid optimizer.")

    def __call__(self, feed_dict, sess, device_string="/cpu:0"):
        with tf.device(device_string):
            sess.run(self.solver_op, feed_dict)


class PolicyOptTf(PolicyOpt):
    """ Policy optimization using tensor flow for DAG computations/nonlinear function approximation. """
    def __init__(self, hyperparams, dO, dU):
        config = copy.deepcopy(POLICY_OPT_TF)
        config.update(hyperparams)

        PolicyOpt.__init__(self, config, dO, dU)

        self.tf_iter = 0
        self.checkpoint_file = self._hyperparams['checkpoint_prefix']
        self.batch_size = self._hyperparams['batch_size']
        self.device_string = "/cpu:0"
        if self._hyperparams['use_gpu']:
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
        self.policy = TfPolicy(self.obs_tensor, self.act_op, np.zeros(dU), self.sess)
        tf.initialize_all_variables()

    def init_network(self):
        """ Helper method to initialize the tf networks used """
        tf_model = self._hyperparams['network_model']
        model_dict = tf_model(dim_input=self._dO, dim_output=self._dU, batch_size=None)
        self.obs_tensor = model_dict['inputs'][0]
        self.action_tensor = model_dict['inputs'][1]
        self.precision_tensor = model_dict['inputs'][2]
        self.act_op = model_dict['outputs'][0]
        self.loss_scalar = model_dict['loss'][0]

    def init_solver(self):
        """ Helper method to initialize the solver. """
        self.solver = TfSolver(loss_scalar=self.loss_scalar,
                               solver_name=self._hyperparams['solver_type'],
                               snapshot_prefix=self._hyperparams['weights_file_prefix'],
                               base_lr=self._hyperparams['lr'],
                               lr_policy=self._hyperparams['lr_policy'],
                               momentum=self._hyperparams['momentum'],
                               weight_decay=self._hyperparams['weight_decay'])

    # TODO - This assumes that the obs is a vector being passed into the
    #        network in the same place.
    #        (won't work with images or multimodal networks)
    def update(self, obs, tgt_mu, tgt_prc, tgt_wt, itr, inner_itr):
        """
        Update policy.
        Args:
            obs: Numpy array of observations, N x T x dO.
            tgt_mu: Numpy array of mean controller outputs, N x T x dU.
            tgt_prc: Numpy array of precision matrices, N x T x dU x dU.
            tgt_wt: Numpy array of weights, N x T.
        Returns:
            A CaffePolicy object with updated weights.
        """
        N, T = obs.shape[:2]
        dU, dO = self._dU, self._dO

        # TODO - Make sure all weights are nonzero?

        # Save original tgt_prc.
        tgt_prc_orig = np.reshape(tgt_prc, [N*T, dU, dU])

        # Renormalize weights.
        tgt_wt *= (float(N * T) / np.sum(tgt_wt))
        # Allow weights to be at most twice the robust median.
        mn = np.median(tgt_wt[(tgt_wt > 1e-2).nonzero()])
        for n in range(N):
            for t in range(T):
                tgt_wt[n, t] = min(tgt_wt[n, t], 2 * mn)
        # Robust median should be around one.
        tgt_wt /= mn

        # Reshape inputs.
        obs = np.reshape(obs, (N*T, dO))
        tgt_mu = np.reshape(tgt_mu, (N*T, dU))
        tgt_prc = np.reshape(tgt_prc, (N*T, dU, dU))
        tgt_wt = np.reshape(tgt_wt, (N*T, 1, 1))

        # Fold weights into tgt_prc.
        tgt_prc = tgt_wt * tgt_prc

        #TODO: Find entries with very low weights?

        # Normalize obs, but only compute normalzation at the beginning.
        if itr == 0 and inner_itr == 1:
            self.policy.scale = np.diag(1.0 / np.std(obs, axis=0))
            self.policy.bias = -np.mean(obs.dot(self.policy.scale), axis=0)
        obs = obs.dot(self.policy.scale) + self.policy.bias

        # Assuming that N*T >= self.batch_size.
        batches_per_epoch = np.floor(N*T / self.batch_size)
        idx = range(N*T)
        average_loss = 0
        np.random.shuffle(idx)
        for i in range(self._hyperparams['iterations']):
            # Load in data for this batch.
            start_idx = int(i * self.batch_size %
                            (batches_per_epoch * self.batch_size))
            idx_i = idx[start_idx:start_idx+self.batch_size]
            feed_dict = {self.obs_tensor: obs[idx_i],
                         self.action_tensor: tgt_mu[idx_i],
                         self.precision_tensor: tgt_prc[idx_i]}
            self.solver(feed_dict, self.sess)
            with tf.device(self.device_string):
                train_loss = self.sess.run(self.loss_scalar, feed_dict)

            average_loss += train_loss
            if i % 500 == 0 and i != 0:
                LOGGER.debug('tensorflow iteration %d, average loss %f',
                             i, average_loss / 500)
                average_loss = 0

        # Keep track of tensorflow iterations for loading solver states.
        self.tf_iter += self._hyperparams['iterations']

        # Optimize variance.
        A = np.sum(tgt_prc_orig, 0) + 2 * N * T * \
                                      self._hyperparams['ent_reg'] * np.ones((dU, dU))
        A = A / np.sum(tgt_wt)

        # TODO - Use dense covariance?
        self.var = 1 / np.diag(A)

        return self.policy

    def prob(self, obs):
        """
        Run policy forward.
        Args:
            obs: Numpy array of observations that is N x T x dO.
        """
        dU = self._dU
        N, T = obs.shape[:2]

        # Normalize obs.
        try:
            for n in range(N):
                obs[n, :, :] = obs[n, :, :].dot(self.policy.scale) + \
                        self.policy.bias
        except AttributeError:
            pass  #TODO: Should prob be called before update?

        output = np.zeros((N, T, dU))

        for i in range(N):
            for t in range(T):
                # Feed in data.
                feed_dict = {self.obs_tensor: obs[i, t]}
                output[i, t, :] = self.sess.run(self.act_op, feed_dict=feed_dict)

        pol_sigma = np.tile(np.diag(self.var), [N, T, 1, 1])
        pol_prec = np.tile(np.diag(1.0 / self.var), [N, T, 1, 1])
        pol_det_sigma = np.tile(np.prod(self.var), [N, T])

        return output, pol_sigma, pol_prec, pol_det_sigma

    def set_ent_reg(self, ent_reg):
        """ Set the entropy regularization. """
        self._hyperparams['ent_reg'] = ent_reg

    # For pickling.
    def __getstate__(self):
        saver = tf.train.Saver()
        saver.save(self.sess, save_path=self.checkpoint_file)
        return {
            'hyperparams': self._hyperparams,
            'dO': self._dO,
            'dU': self._dU,
            'scale': self.policy.scale,
            'bias': self.policy.bias,
            'tf_iter': self.tf_iter,
        }

    # For unpickling.
    def __setstate__(self, state):
        self.__init__(state['hyperparams'], state['dO'], state['dU'])
        self.policy.scale = state['scale']
        self.policy.bias = state['bias']
        self.tf_iter = state['tf_iter']

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_file)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise IOError('checkpoint file not found.')
