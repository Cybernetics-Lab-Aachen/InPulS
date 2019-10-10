""" This file defines an agent for the Kinova Jaco2 ROS environment. """
import copy
import logging
import time
import numpy as np

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_PANDA
from gps.agent.panda.utils import TimeoutException, ServiceEmulator, msg_to_sample

from gps.proto.gps_pb2 import TRIAL_ARM, ACTION, END_EFFECTOR_POINT_JACOBIANS

import gps.proto.Command_pb2 as command_msgs

from gps.sample.sample import Sample

# # Approximation of Jaco EE Jacobian
# JAC = np.array(
#     [
#         -0.29052, -0.0456493, -0.232069, 0.0490652, 0.0297493, 0, 0.294644, -0.0288231, -0.136461, 0.0604075,
#         -0.00100644, 0, 6.09391e-14, -0.402527, 0.156064, -0.0512757, 0.016437, 0, 0, 0.508486, -0.508486, 0.486912,
#         -0.401568, -0.0127389, 2.06823e-13, -0.856, 0.856, 0.292873, 0.509841, 0.99136, -1, -1.76985e-13, 3.01313e-13,
#         0.814991, 0.759208, 0.0832759
#     ]
# ).reshape(6, 6)


class AgentPanda(Agent):
    """
    All communication between the algorithms and ROS is done through
    this class.
    """

    def __init__(self, hyperparams, init_node=True):
        """
        Initialize agent.
        Args:
            hyperparams: Dictionary of hyperparameters.
            init_node: Whether or not to initialize a new ROS node.
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

        #self.scaler = self._hyperparams.get('scaler', None)

        # # EE Jacobian
        # self.jac = JAC
        # if self.scaler is not None:  # Scale jacobians
        #     self.jac[:3] *= self.scaler.scale_[:6].reshape(1, 6)
        #     self.jac[:3] /= self.scaler.scale_[-9:-6].reshape(3, 1)

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

    def get_data(self, **kwargs):
        """
        Request for the most recent value for data/sensor readings.
        Returns entire sample report (all available data) in sample.
        Args:
        """
        request = command_msgs.Request()
        request.id = self._get_next_seq_id()
        request.ee_offsets.extend(self.ee_points.reshape(-1))
        result_msg = self._data_service.publish_and_wait(request, poll_delay=0.001)
        # TODO - Make IDs match, assert that they match elsewhere here.
        return msg_to_sample(result_msg, self)

    def reset_arm(self, arm, mode, data, position_command=False):
        """
        Issues a position command to an arm.
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

    def relax_arm(self, *args, **kwargs):
        reset_command = command_msgs.Command()
        reset_command.command.extend([0., 0., 0., 0., 0., 0., 0.])
        reset_command.is_position_command = False
        reset_command.ee_offsets.extend(self.ee_points.reshape(-1))
        self._trial_service.publish(reset_command)

    def reset(self, condition):
        """
        Reset the agent for a particular experiment condition.
        Args:
            condition: An index into hyperparams['reset_conditions'].
        """
        condition_data = self._hyperparams['reset_conditions'][condition]
        if condition is None:
            raise NotImplementedError()
        else:
            self.reset_arm(None, None, condition_data[TRIAL_ARM]['data'], True)
        time.sleep(0.5)  # useful for the real robot, so it stops completely

    # def transform(self, sensor_type, data):
    #     idx = self._x_data_idx[sensor_type]
    #     return (data - self.scaler.mean_[idx]) / self.scaler.scale_[idx]

    def sample(self, policy, condition, save=True, noisy=True, reset_cond=None, **kwargs):
        """
        Reset and execute a policy and collect a sample.
        Args:
            policy: A Policy object.
            condition: Which condition setup to run.
            save: Whether or not to store the trial into the samples.
            noisy: Whether or not to use noise during sampling.
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
        """ relax arm to change mode to torque. If this is not done, the mode will be changed in timestep t=0 causing
        the loop to be slow in timestep t=1 because the mutex in the cpp is locked. """
        self.relax_arm()
        time.sleep(1)

        start = time.time()
        for t in range(self.T):
            # Read sensors and store sensor data in sample
            latest_sample = self.get_data()
            for sensor_type in self.x_data_types:
                data = latest_sample.get(sensor_type)
                # if self.scaler is not None:
                #     data = self.transform(sensor_type, data)
                sample.set(sensor_type, data, t)
            sample.set(END_EFFECTOR_POINT_JACOBIANS, latest_sample.get(END_EFFECTOR_POINT_JACOBIANS), t=t)
            # position = latest_sample.get(END_EFFECTOR_POINTS)
            # diff = np.array(self.ee_points_tgt - position).reshape(3, 3)
            # norm = np.linalg.norm(diff, axis=1)
            # print(norm)

            # Use END_EFFECTOR_POINTS as distance to target
            # sample.set(
            #     END_EFFECTOR_POINTS,
            #     sample.get(END_EFFECTOR_POINTS, t),# - self.ee_points_tgt / self.scaler.scale_[-9:],
            #     t=t
            # )

            # Get action
            U_t = policy.act(sample.get_X(t), sample.get_obs(t), t, noise)
            torque_limits_ = np.array([4.0, 4.0, 4.0, 4.0, 1.0, 1.0,
                                       .5])  # TODO: find better solution to clip (same as in cpp)
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
