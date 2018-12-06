""" This file defines an agent for the Kinova Jaco2 ROS environment. """
import copy
import numpy as np
import logging
import time

import gym

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.sample.sample import Sample
from gps.proto.gps_pb2 import ACTION, END_EFFECTOR_POINTS


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
        self.__init_gym() # TODO Emma only for testing!

    def __init_gym(self):
        import types

        self.env = gym.make(self._hyperparams['env'])
        self.sim = self.env.env.sim
        self.env.env.render = types.MethodType(render, self.env.env)  # Dirty hack to work around the fixed 500x500 pixel recording
        self.env._max_episode_steps = self.T-1 # So env is done with the last timestep
        self.env = gym.wrappers.Monitor(self.env , self._hyperparams['data_files_dir'], force=True)
        if is_goal_based(self.env):
            dX = self.env.observation_space.spaces['observation'].shape[0] + self.env.observation_space.spaces['desired_goal'].shape[0] 
        else:
            dX = self.env.observation_space.shape[0]
        dU = self.env.action_space.shape[0]

        #data = self.sim.data
        #print("qpos", data.qpos.flat[1:].shape)
        #print("qvel", data.qvel.flat[:].shape)
        #exit(0)

        assert self.dX == dX, 'expected dX=%d, got dX=%d'%(self.dX, dX)
        assert self.dU == dU, 'expected dU=%d, got dU=%d'%(self.dU, dU)

    def sample(self, policy, condition, verbose=True, save=True, noisy=True,
               use_TfController=False, first_itr=False, timeout=None, reset=True, rnd=False, record=False):
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
        if not rnd:
            self.env.seed(self.x0[condition])
        obs = self.env.reset() 
        self.set_states(sample, obs, 0)
        #sample.set('observation', obs['observation'], 0)
        #sample.set(END_EFFECTOR_POINTS, np.asarray(obs['desired_goal']) - np.asarray(obs['achieved_goal']), 0)
        U_0 = policy.act(sample.get_X(0), sample.get_obs(0), 0, noise)
        sample.set(ACTION, U_0, 0)
        for t in range(1, self.T):
            if not record and self.render:
                self.env.render(mode='human')  # TODO add hyperparam

            # Get state
            obs, _, done, _ = self.env.step(sample.get_U(t-1))
            self.set_states(sample, obs, t)

            # Get action
            U_t = policy.act(sample.get_X(t), sample.get_obs(t), t, noise)
            sample.set(ACTION, U_t, t)

            if done and t < self.T-1:
                raise Exception('Iteration ended prematurely %d/%d' %(t+1, self.T))
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
            sample.set(END_EFFECTOR_POINTS, np.asarray(obs['desired_goal']) - np.asarray(obs['achieved_goal']), t)  # Use goal as EE
            obs = obs['observation']
        else:
            sample.set(END_EFFECTOR_POINTS, self.sim.data.qpos[:2], t)  # Use first pos as EE TODO Depends on env

        for data_type in self._x_data_idx:
            if data_type != END_EFFECTOR_POINTS:
                sample.set(data_type, obs[self._x_data_idx[data_type]], t)


def is_goal_based(env):
    return isinstance(env.observation_space, gym.spaces.Dict)

def render(self, mode='human'):
    """
    Dirty hack to work around the fixed 500x500 pixel recording in gym.
    """
    self._render_callback()
    if mode == 'rgb_array':
        self._get_viewer().render()
        return self._get_viewer()._read_pixels_as_in_window()
    elif mode == 'human':
        self._get_viewer().render()
