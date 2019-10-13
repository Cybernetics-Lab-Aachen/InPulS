"""This module defines an agent for the Franka Emika Panda robot arm."""
import copy
import logging
import time
import numpy as np

from gps.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_PANDA
from gps.agent.ros_utils import TimeoutException, ServiceEmulator
from gps.sample import Sample
from gps.proto.gps_pb2 import (
    TRIAL_ARM, ACTION, END_EFFECTOR_POINT_JACOBIANS, END_EFFECTOR_POINTS, JOINT_ANGLES, JOINT_VELOCITIES
)
import gps.proto.Command_pb2 as command_msgs


class AgentPanda(Agent):
    """An Agent for the Franka Emika Panda robot arm."""

    def __init__(self, hyperparams):
        """Initializes the agent.

        Args:
            hyperparams: Dictionary of hyperparameters.

        """
        config = copy.deepcopy(AGENT_PANDA)
        config.update(hyperparams)
        Agent.__init__(self, config)
        self._init_pubs_and_subs()
        self._seq_id = 0  # Used for setting seq in ROS commands.

        conditions = self._hyperparams['conditions']

        self.x0 = []
        for field in ('x0', 'ee_points_tgt', 'reset_conditions'):
            self._hyperparams[field] = setup(self._hyperparams[field], conditions)
        self.x0 = self._hyperparams['x0']
        self.dt = self._hyperparams['dt']

        self.ee_points = hyperparams["ee_points"]
        self.ee_points_tgt = self._hyperparams['ee_points_tgt']

    def _init_pubs_and_subs(self):
        self._trial_service = ServiceEmulator(
            self._hyperparams['trial_command_topic'], command_msgs.Command, self._hyperparams['sample_result_topic'],
            command_msgs.State, self._hyperparams["trial_pub_url"], self._hyperparams["sub_url"]
        )

        self._data_service = ServiceEmulator(
            self._hyperparams['data_request_topic'], command_msgs.Request, self._hyperparams['sample_result_topic'],
            command_msgs.State, self._hyperparams["request_pub_url"], self._hyperparams["sub_url"]
        )

        time.sleep(1)  # wait for sub/pub to set up before publishing a message

    def _get_next_seq_id(self):
        self._seq_id = (self._seq_id + 1) % (2**30)
        return self._seq_id

    def get_data(self):
        """Requests the most recent value for data/sensor readings.

        Returns:
            sample: entire sample report (all available data).

        """
        request = command_msgs.Request()
        request.id = self._get_next_seq_id()
        request.ee_offsets.extend(self.ee_points.reshape(-1))
        result_msg = self._data_service.publish_and_wait(request, poll_delay=0.001)
        # TODO - Make IDs match, assert that they match elsewhere here.
        return msg_to_sample(result_msg, self)

    def reset_arm(self, arm, mode, data, position_command=False):
        """Issues a position command to an arm.

        Args:
            arm: Either TRIAL_ARM or AUXILIARY_ARM.
            mode: An integer code (defined in gps_pb2).
            data: An array of floats.

        """
        reset_command = command_msgs.Command()
        reset_command.command.extend(data)
        reset_command.is_position_command = position_command
        reset_command.ee_offsets.extend(self.ee_points.reshape(-1))
        if position_command:
            try:
                self._trial_service.publish_and_wait(
                    reset_command,
                    timeout=self._hyperparams['trial_timeout'],
                )
            except TimeoutException:
                input('The robot arm seems to be stuck. Unstuck it and press <ENTER> to continue.')
                self.reset_arm(arm, mode, data, position_command)
        else:
            self._trial_service.publish(reset_command)

    def relax_arm(self):
        """Relaxes current arm."""
        reset_command = command_msgs.Command()
        reset_command.command.extend([0., 0., 0., 0., 0., 0., 0.])
        reset_command.is_position_command = False
        reset_command.ee_offsets.extend(self.ee_points.reshape(-1))
        self._trial_service.publish(reset_command)

    def reset(self, condition):
        """Reset the agent for a particular experiment condition.

        Args:
            condition: An index into hyperparams['reset_conditions'].

        """
        condition_data = self._hyperparams['reset_conditions'][condition]
        if condition is None:
            raise NotImplementedError()
        else:
            self.reset_arm(None, None, condition_data[TRIAL_ARM]['data'], True)
        time.sleep(0.5)  # useful for the real robot, so it stops completely

    def sample(self, policy, condition, save=True, noisy=True, reset_cond=None, **kwargs):
        """Performs agent reset and rolls out given policy to collect a sample.

        Args:
            policy: A Policy object.
            condition: Which condition setup to run.
            save: Whether or not to store the trial into the samples.
            noisy: Whether or not to use noise during sampling.
            reset_cond: The initial condition to reset the agent into.

        Returns:
            sample: A Sample object.

        """
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))

        sample = Sample(self)
        self.reset(reset_cond)

        # Execute policy over a time period of [0,T]
        # TODO: Find better solution to change mode.
        # relax arm to change mode to torque. If this is not done, the mode will be changed in timestep t=0 causing
        # the loop to be slow in timestep t=1 because the mutex in the cpp is locked. """
        self.relax_arm()
        time.sleep(1)

        start = time.time()
        for t in range(self.T):
            # Read sensors and store sensor data in sample
            latest_sample = self.get_data()
            for sensor_type in self.x_data_types:
                sample.set(sensor_type, latest_sample.get(sensor_type), t)
            sample.set(END_EFFECTOR_POINT_JACOBIANS, latest_sample.get(END_EFFECTOR_POINT_JACOBIANS), t=t)

            # Get action
            U_t = policy.act(sample.get_X(t), sample.get_obs(t), t, noise)

            # TODO: find better solution to clip (same as in cpp)
            torque_limits_ = np.array([4.0, 4.0, 4.0, 4.0, 1.0, 1.0, .5])
            U_t = np.clip(U_t, -torque_limits_, torque_limits_)

            # Perform action
            self.reset_arm(None, None, U_t, False)
            sample.set(ACTION, U_t, t)

            # Check if agent is keeping up
            sleep_time = start + (t + 1) * self.dt - time.time()
            if sleep_time < 0:
                logging.critical("Agent can't keep up.In timestep %i it is %fs behind." % (t, sleep_time))
            elif sleep_time < self.dt / 2:
                logging.warning(
                    "Agent may not keep up (%.0f percent busy)" % (((self.dt - sleep_time) / self.dt) * 100)
                )

            # Wait for next timestep
            if sleep_time > 0:
                time.sleep(sleep_time)

        if save:
            self._samples[condition].append(sample)
        self.reset(reset_cond)
        return sample


def msg_to_sample(ros_msg, agent):
    """Convert a SampleResult ROS message into a Sample Python object."""
    sample = Sample(agent)

    velocity = np.array(ros_msg.velocity).reshape(7)
    joint_angles = np.array(ros_msg.joint_angles).reshape(7)
    ee_pos = np.array(ros_msg.ee_pos).reshape(9)
    ee_jacobians = np.array(ros_msg.ee_points_jacobian, order="F").reshape(9, 7)

    sample.set(JOINT_VELOCITIES, velocity)
    sample.set(JOINT_ANGLES, joint_angles)
    sample.set(END_EFFECTOR_POINTS, ee_pos)
    sample.set(END_EFFECTOR_POINT_JACOBIANS, ee_jacobians)

    return sample
