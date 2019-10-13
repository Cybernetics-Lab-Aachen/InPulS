import numpy as np
from baselines.common import set_global_seeds
from baselines.common.tf_util import get_session
from baselines.ddpg.ddpg_learner import DDPG
from baselines.ddpg.memory import Memory
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.noise import AdaptiveParamNoiseSpec
from tqdm import tqdm

from gps.algorithm.policy_opt.policy_opt import PolicyOpt


class DDPG_Policy(PolicyOpt):
    def __init__(self, hyperparams, dX, dU):
        PolicyOpt.__init__(self, hyperparams, dX, dU)
        self.dX = dX
        self.dU = dU

        self.epochs = hyperparams['epochs']
        self.param_noise_adaption_interval = hyperparams['param_noise_adaption_interval']
        set_global_seeds(hyperparams['seed'])

        # Initialize DDPG policy
        self.pol = DDPG(
            Actor(dU, network=hyperparams['network'], **hyperparams['network_kwargs']),
            Critic(network=hyperparams['network'], **hyperparams['network_kwargs']),
            Memory(limit=hyperparams['memory_limit'], action_shape=(dU, ), observation_shape=(dX, )),
            observation_shape=(dX, ),
            action_shape=(dU, ),
            param_noise=AdaptiveParamNoiseSpec(initial_stddev=0.2, desired_action_stddev=0.2),
            **hyperparams['ddpg_kwargs']
        )

        sess = get_session()
        self.pol.initialize(sess)
        sess.graph.finalize()

        self.policy = self  # Act method is contained in this class

    def update(self, X, U, cs, **kwargs):
        M, N, T, _ = X.shape

        # Store samples in memory
        self.pol.store_transition(
            X[:, :, :-1].reshape(M * N * (T - 1), self.dX),
            U[:, :, :-1].reshape(M * N * (T - 1), self.dU),
            -cs[:, :, :-1].reshape(M * N * (T - 1)),
            X[:, :, 1:].reshape(M * N * (T - 1), self.dX),
            np.zeros((M * N * (T - 1))),
        )

        # Train DDPG
        losses = np.zeros((self.epochs, 2))
        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            if self.pol.memory.nb_entries >= self.pol.batch_size and epoch % self.param_noise_adaption_interval == 0:
                self.pol.adapt_param_noise()

            losses[epoch] = self.pol.train()
            self.pol.update_target_net()

            pbar.set_description("Loss: %.6f/%.6f" % (losses[epoch, 0], losses[epoch, 1]))

        # Visualize training loss
        from gps.visualization import visualize_loss
        visualize_loss(
            self._data_files_dir + 'plot_gps_training-%02d' % (self.iteration_count),
            losses,
            labels=['critic', 'actor'],
        )

    def act(self, x, _, t, noise):
        """Decides an action for the given state/observation at the current timestep.

        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: A dU-dimensional noise vector.

        Returns:
            A dU dimensional action vector.

        """
        if t == 0:
            self.pol.reset()

        u = self.pol.step(x, apply_noise=np.any(noise), compute_Q=False)[0][0]
        return u
