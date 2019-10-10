""" This file defines an agent for the Kinova Jaco2 ROS environment. """
import numpy as np

import gym

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise
from gps.sample.sample import Sample
from gps.proto.gps_pb2 import ACTION


class AgentOpenAIGym(Agent):
    """
    All communication between the algorithms and ROS is done through
    this class.
    """

    def __init__(self, hyperparams):
        """
        Initialize agent.
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
        #self.env.render(mode='human')  # Render once to init opengl TODO: Add video capute hyperparam

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
        verbose=True,
        save=True,
        noisy=True,
        use_TfController=False,
        timeout=None,
        reset_cond=None,
        randomize_initial_state=0,
        record=False
    ):
        """
        Reset and execute a policy and collect a sample.
        Args:
            policy: A Policy object.
            condition: Which condition setup to run.
            verbose: Unused for this agent.
            save: Whether or not to store the trial into the samples.
            noisy: Whether or not to use noise during sampling.
            use_TfController: Whether to use the syncronous TfController
        Returns:
            sample: A Sample object.
        """

        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))

        # Get a new sample
        sample = Sample(self)

        self.env.video_callable = lambda episode_id, record=record: record
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

        if record:
            from gym.wrappers.monitoring.video_recorder import ImageEncoder

            rgb = self.env.render(mode='rgb_array')
            encoder = ImageEncoder(
                self._hyperparams['data_files_dir'] + capture_name + '.mp4', rgb.shape, 1 / self._hyperparams['dt']
            )
            encoder.capture_frame(rgb)

        self.set_states(sample, obs, 0)
        U_0 = policy.act(sample.get_X(0), sample.get_obs(0), 0, noise)
        sample.set(ACTION, U_0, 0)
        for t in range(1, self.T):
            if self.render:
                self.env.render(mode='human')
            if record:
                encoder.capture_frame(self.env.render(mode='rgb_array'))

            # Get state
            obs, _, done, _ = self.env.step(sample.get_U(t - 1))
            self.set_states(sample, obs, t)

            # Get action
            U_t = policy.act(sample.get_X(t), sample.get_obs(t), t, noise, self.action_noise_clip)
            sample.set(ACTION, U_t, t)

            if done and t < self.T - 1:
                raise Exception('Iteration ended prematurely %d/%d' % (t + 1, self.T))
        if record:
            encoder.close()
        if save:
            self._samples[condition].append(sample)
        self.active = False
        #print("X", sample.get_X())
        #print("U", sample.get_U())
        return sample

    def set_states(self, sample, obs, t):
        """
        Reads individual sensors from obs and store them in the sample.
        """
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
    return isinstance(env.observation_space, gym.spaces.Dict)
