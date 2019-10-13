"""This module defines an agent for gym environments."""
import numpy as np

import gym

from gps.agent import Agent
from gps.agent.agent_utils import generate_noise
from gps.sample.sample import Sample
from gps.proto.gps_pb2 import ACTION


class AgentOpenAIGym(Agent):
    """An Agent for gym environments."""

    def __init__(self, hyperparams):
        """Initializes the agent.

        Args:
            hyperparams: Dictionary of hyperparameters.

        """
        Agent.__init__(self, hyperparams)
        self.x0 = self._hyperparams['x0']
        self.record = False
        self.render = self._hyperparams['render']
        self.scaler = self._hyperparams.get('scaler', None)
        self.action_noise_clip = self._hyperparams.get('action_noise_clip', None)
        self.__init_gym()

    def __init_gym(self):
        self.env = gym.make(self._hyperparams['env'])
        if isinstance(self.env, gym.wrappers.TimeLimit):
            self.env = self.env.env

        self.sim = self.env.sim
        if is_goal_based(self.env):
            dX = (
                self.env.observation_space.spaces['observation'].shape[0] +
                self.env.observation_space.spaces['desired_goal'].shape[0]
            )
        else:
            dX = self.env.observation_space.shape[0]
        dU = self.env.action_space.shape[0]

        assert self.dX == dX, 'expected dX=%d, got dX=%d' % (self.dX, dX)
        assert self.dU == dU, 'expected dU=%d, got dU=%d' % (self.dU, dU)

    def sample(
        self,
        policy,
        condition,
        save=True,
        noisy=True,
        reset_cond=None,
        randomize_initial_state=0,
        **kwargs,
    ):
        """Performs agent reset and rolls out given policy to collect a sample.

        Args:
            policy: A Policy object.
            condition: Which condition setup to run.
            save: Whether or not to store the trial into the samples.
            noisy: Whether or not to use noise during sampling.
            reset_cond: The initial condition to reset the agent into.
            randomize_initial_state: Perform random steps after resetting to simulate a noisy initial state.

        Returns:
            sample: A Sample object.

        """
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))

        # Get a new sample
        sample = Sample(self)

        # Get initial state
        self.env.seed(None if reset_cond is None else self.x0[reset_cond])
        obs = self.env.reset()
        if randomize_initial_state > 0:
            # Take random steps randomize initial state distribution
            self.env._set_action(
                (self.env.action_space.high - self.env.action_space.low) / 12 * np.random.normal(size=self.dU) *
                randomize_initial_state
            )
            for _ in range(5):
                self.sim.step()
            obs = self.env.step(np.zeros(self.dU))[0]

        self.set_states(sample, obs, 0)
        U_0 = policy.act(sample.get_X(0), sample.get_obs(0), 0, noise)
        sample.set(ACTION, U_0, 0)
        for t in range(1, self.T):
            if self.render:
                self.env.render(mode='human')

            # Get state
            obs, _, done, _ = self.env.step(sample.get_U(t - 1))
            self.set_states(sample, obs, t)

            # Get action
            U_t = policy.act(sample.get_X(t), sample.get_obs(t), t, noise, self.action_noise_clip)
            sample.set(ACTION, U_t, t)

            if done and t < self.T - 1:
                raise Exception('Iteration ended prematurely %d/%d' % (t + 1, self.T))
        if save:
            self._samples[condition].append(sample)
        return sample

    def set_states(self, sample, obs, t):
        """Reads individual sensors from obs and store them in the sample."""
        if is_goal_based(self.env):
            X = np.concatenate([obs['observation'], np.asarray(obs['desired_goal']) - np.asarray(obs['achieved_goal'])])
        else:
            X = obs

        # Scale states
        if self.scaler:
            X = self.scaler.transform([X])[0]

        for sensor, idx in self._x_data_idx.items():
            sample.set(sensor, X[idx], t)

        if 'additional_sensors' in self._hyperparams:
            self._hyperparams['additional_sensors'](self.sim, sample, t)


def is_goal_based(env):
    """Determines wherter a gym environemtn is goal based, i.e. supplies `desired_goal` and `achieved_goal`."""
    return isinstance(env.observation_space, gym.spaces.Dict)
