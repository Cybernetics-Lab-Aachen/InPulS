""" This file defines policy optimization for a tensorflow policy. """
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from gps.algorithm.policy_opt.policy_opt import PolicyOpt


class GPS_Policy(PolicyOpt):
    """ Policy optimization using tensor flow for DAG computations/nonlinear function approximation. """

    def __init__(self, hyperparams, dX, dU):
        PolicyOpt.__init__(self, hyperparams, dX, dU)
        self.dX = dX
        self.dU = dU

        tf.set_random_seed(self._hyperparams['random_seed'])
        self.var = self._hyperparams['init_var'] * np.ones(dU)
        self.epochs = self._hyperparams['epochs']
        self.batch_size = self._hyperparams['batch_size']
        self.weight_decay = self._hyperparams['weight_decay']
        self.N_hidden = self._hyperparams['N_hidden']

        self.graph = tf.Graph()  # Encapsulate model in own graph
        with self.graph.as_default():
            self.init_network()
            self.init_loss_function()
            self.init_solver()

            # Create session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True  # Prevent GPS from hogging all memory
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

        self.policy = self  # Act method is contained in this class
        self.scaler = None

    def init_network(self):
        # Placeholders for dataset
        self.state_data = tf.placeholder(tf.float32, (None, self.dX))
        self.action_data = tf.placeholder(tf.float32, (None, self.dU))
        self.precision_data = tf.placeholder(tf.float32, (None, self.dU, self.dU))
        dataset = tf.data.Dataset.from_tensor_slices((
            self.state_data,
            self.action_data,
            self.precision_data,
        )).shuffle(10000).batch(self.batch_size).repeat()

        # Batch iterator
        self.iterator = dataset.make_initializable_iterator()
        self.state_batch, self.action_batch, self.precision_batch = self.iterator.get_next()

        with arg_scope(
            [layers.fully_connected],
            activation_fn=tf.nn.relu,
            weights_regularizer=layers.l2_regularizer(scale=self.weight_decay)
        ):
            h = layers.fully_connected(self.state_batch, self.N_hidden)
            h = layers.fully_connected(h, self.N_hidden)
            h = layers.fully_connected(h, self.N_hidden)
        self.action_out = layers.fully_connected(h, self.dU, activation_fn=None)

    def init_loss_function(self):
        # KL divergence loss
        #  loss_kl = 1/2 delta_action^T * prc * delta_action
        delta_action = self.action_batch - self.action_out
        self.loss_kl = tf.reduce_mean(tf.einsum('in,inm,im->i', delta_action, self.precision_batch, delta_action)) / 2

        # Regularization loss
        self.loss_reg = tf.losses.get_regularization_loss()

        # Total loss
        self.loss = self.loss_kl + self.loss_reg

    def init_solver(self):
        optimizer = tf.train.AdamOptimizer()
        self.solver_op = optimizer.minimize(self.loss)
        self.optimizer_reset_op = tf.variables_initializer(optimizer.variables())

    def update(self, X, mu, prc, _):
        """
        Trains a GPS model on the dataset
        """
        N, T, _ = X.shape

        # Reshape inputs.
        X = X.reshape((N * T, self.dX))
        mu = mu.reshape((N * T, self.dU))
        prc = prc.reshape((N * T, self.dU, self.dU))

        # Normalize X, but only compute normalization at the beginning.
        if self.scaler is None:
            self.scaler = StandardScaler().fit(X)
        X = self.scaler.transform(X)

        # Normalize precision
        prc = prc * (self.dU / np.mean(np.trace(prc, axis1=-2, axis2=-1)))

        # Reset optimizer
        self.sess.run(self.optimizer_reset_op)

        # Initialize dataset iterator
        self.sess.run(
            self.iterator.initializer, feed_dict={
                self.state_data: X,
                self.action_data: mu,
                self.precision_data: prc
            }
        )

        batches_per_epoch = int(N * T / self.batch_size)
        assert batches_per_epoch * self.batch_size == N * T, \
            '%d * %d != %d * %d' % (batches_per_epoch, self.batch_size, N, T)
        losses = np.zeros((self.epochs, 2))
        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            for i in range(batches_per_epoch):
                losses[epoch] += self.sess.run([
                    self.solver_op,
                    self.loss_kl,
                    self.loss_reg,
                ])[1:]
            losses[epoch] /= batches_per_epoch
            pbar.set_description("GPS Loss: %.6f" % (np.sum(losses[epoch])))

        # Visualize training loss
        from gps.visualization import visualize_loss
        visualize_loss(
            self._data_files_dir + 'plot_gps_training-%02d' % (self.iteration_count),
            losses,
            labels=['KL divergence', 'L2 reg']
        )

        # Optimize variance.
        A = np.mean(prc, axis=0) + 2 * N * T * self._hyperparams['ent_reg'] * np.ones((self.dU, self.dU))

        self.var = 1 / np.diag(A)
        self.policy.chol_pol_covar = np.diag(np.sqrt(self.var))

    def act(self, x, _, t, noise):
        u = self.sess.run(
            self.action_out, feed_dict={
                self.state_batch: self.scaler.transform([x]),
            }
        )[0]
        if noise is not None:
            if t is None:
                u += self.chol_pol_covar.T.dot(noise[0])
            else:
                u += self.chol_pol_covar.T.dot(noise[t])
        return u

    def prob(self, X):
        """
        Run policy forward.
        Args:
            X: States (N, T, dX)
        """
        N, T = X.shape[:2]

        action = self.sess.run(
            self.action_out, feed_dict={
                self.state_batch: self.scaler.transform(X.reshape(N * T, self.dX))
            }
        ).reshape((N, T, self.dU))
        pol_sigma = np.tile(np.diag(self.var), [N, T, 1, 1])
        pol_prec = np.tile(np.diag(1.0 / self.var), [N, T, 1, 1])
        pol_det_sigma = np.tile(np.prod(self.var), [N, T])

        return action, pol_sigma, pol_prec, pol_det_sigma
