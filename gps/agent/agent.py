"""This file defines the base agent class."""
from abc import ABC, abstractmethod
import copy

from gps.agent.config import AGENT
from gps.proto.gps_pb2 import ACTION
from gps.sample import Sample, SampleList


class Agent(ABC):
    """Abstract agent superclass. The agent interacts with the environment to collect samples."""

    def __init__(self, hyperparams):
        """Initializes the agent.

        Args:
            hyperparams: Dictionary of hyperparameters.

        """
        config = copy.deepcopy(AGENT)
        config.update(hyperparams)
        self._hyperparams = config

        # Store samples, along with size/index information for samples.
        self._samples = [[] for _ in range(self._hyperparams['conditions'])]
        self.T = self._hyperparams['T']

        self.x_data_types = self._hyperparams['state_include']
        self.obs_data_types = self._hyperparams['obs_include']
        self.u_data_types = self._hyperparams['actions_include']
        self.meta_data_types = self._hyperparams['meta_include'] if 'meta_include' in self._hyperparams else []
        self.sensor_dims = self._hyperparams['sensor_dims']

        # Construct indices for composite data
        def data_idx(data_types, offset=0):
            idx, i = [], 0
            for sensor in data_types:
                if sensor not in self.sensor_dims:
                    raise ValueError('No sensor dimension for sensor %r' % sensor)
                dim = self.sensor_dims[sensor]
                idx.append(list(range(offset + i, offset + i + dim)))
                i += dim
            data_idx = {d: i for d, i in zip(data_types, idx)}
            return i, data_idx

        self.dX, self._x_data_idx = data_idx(self.x_data_types)
        self.dO, self._obs_data_idx = data_idx(self.obs_data_types)
        self.dM, self._meta_data_idx = data_idx(self.meta_data_types)
        self.dU, self._u_data_idx = data_idx(self.u_data_types)

        self._target_ja = []
        self._initial_ja = []

    @abstractmethod
    def sample(self, policy, condition, save=True, noisy=True, reset_cond=None, **kwargs):
        """Performs agent reset and rolls out given policy to collect a sample.

        Args:
            policy: A Policy object.
            condition: Which condition setup to run.
            save: Whether or not to store the trial into the samples.
            noisy: Whether or not to use noise during sampling.
            reset_cond: The initial condition to reset the agent into.
            kwargs: Additional agent specific arguments.

        Returns:
            sample: A Sample object.

        """
        pass

    def reset(self, condition):
        """Reset the agent for a particular experiment condition.

        Args:
            condition: An index into hyperparams['reset_conditions']. Some environments support `None` to indicate a
            reset to a random initial state.

        """
        pass

    def get_samples(self, condition, start=0, end=None):
        """Returns the requested samples based on the start and end indices.

        Args:
            start: Starting index of samples to return.
            end: End index of samples to return.

        """
        return (
            SampleList(self._samples[condition][start:])
            if end is None else SampleList(self._samples[condition][start:end])
        )

    def pack_data_obs(self, existing_mat, data_to_insert, data_types, axes=None):
        """Updates the observation matrix with new data.

        Args:
            existing_mat: Current observation matrix.
            data_to_insert: New data to insert into the existing matrix.
            data_types: Name of the sensors to insert data for.
            axes: Which axes to insert data. Defaults to the last axes.

        """
        num_sensor = len(data_types)
        if axes is None:
            # If axes not specified, assume indexing on last dimensions.
            axes = list(range(-1, -num_sensor - 1, -1))
        else:
            # Make sure number of sensors and axes are consistent.
            if num_sensor != len(axes):
                raise ValueError('Length of sensors (%d) must equal length of axes (%d)', num_sensor, len(axes))

        # Shape checks.
        insert_shape = list(existing_mat.shape)
        for i in range(num_sensor):
            # Make sure to slice along X.
            if existing_mat.shape[axes[i]] != self.dO:
                raise ValueError('Axes must be along an dX=%d dimensional axis', self.dO)
            insert_shape[axes[i]] = len(self._obs_data_idx[data_types[i]])
        if tuple(insert_shape) != data_to_insert.shape:
            raise ValueError('Data has shape %s. Expected %s', data_to_insert.shape, tuple(insert_shape))

        # Actually perform the slice.
        index = [slice(None) for _ in range(len(existing_mat.shape))]
        for i in range(num_sensor):
            index[axes[i]] = slice(self._obs_data_idx[data_types[i]][0], self._obs_data_idx[data_types[i]][-1] + 1)
        existing_mat[tuple(index)] = data_to_insert

    def pack_data_meta(self, existing_mat, data_to_insert, data_types, axes=None):
        """Updates the meta data matrix with new data.

        Args:
            existing_mat: Current meta data matrix.
            data_to_insert: New data to insert into the existing matrix.
            data_types: Name of the sensors to insert data for.
            axes: Which axes to insert data. Defaults to the last axes.

        """
        num_sensor = len(data_types)
        if axes is None:
            # If axes not specified, assume indexing on last dimensions.
            axes = list(range(-1, -num_sensor - 1, -1))
        else:
            # Make sure number of sensors and axes are consistent.
            if num_sensor != len(axes):
                raise ValueError('Length of sensors (%d) must equal length of axes (%d)', num_sensor, len(axes))

        # Shape checks.
        insert_shape = list(existing_mat.shape)
        for i in range(num_sensor):
            # Make sure to slice along X.
            if existing_mat.shape[axes[i]] != self.dM:
                raise ValueError('Axes must be along an dX=%d dimensional axis', self.dM)
            insert_shape[axes[i]] = len(self._meta_data_idx[data_types[i]])
        if tuple(insert_shape) != data_to_insert.shape:
            raise ValueError('Data has shape %s. Expected %s', data_to_insert.shape, tuple(insert_shape))

        # Actually perform the slice.
        index = [slice(None) for _ in range(len(existing_mat.shape))]
        for i in range(num_sensor):
            index[axes[i]] = slice(self._meta_data_idx[data_types[i]][0], self._meta_data_idx[data_types[i]][-1] + 1)
        existing_mat[index] = data_to_insert

    def pack_data_x(self, existing_mat, data_to_insert, data_types, axes=None):
        """Update the state matrix with new data.

        Args:
            existing_mat: Current state matrix.
            data_to_insert: New data to insert into the existing matrix.
            data_types: Name of the sensors to insert data for.
            axes: Which axes to insert data. Defaults to the last axes.

        """
        num_sensor = len(data_types)
        if axes is None:
            # If axes not specified, assume indexing on last dimensions.
            axes = list(range(-1, -num_sensor - 1, -1))
        else:
            # Make sure number of sensors and axes are consistent.
            if num_sensor != len(axes):
                raise ValueError('Length of sensors (%d) must equal length of axes (%d)', num_sensor, len(axes))

        # Shape checks.
        insert_shape = list(existing_mat.shape)
        for i in range(num_sensor):
            # Make sure to slice along X.
            if existing_mat.shape[axes[i]] != self.dX:
                raise ValueError('Axes must be along an dX=%d dimensional axis', self.dX)
            insert_shape[axes[i]] = len(self._x_data_idx[data_types[i]])
        if tuple(insert_shape) != data_to_insert.shape:
            raise ValueError('Data has shape %s. Expected %s', data_to_insert.shape, tuple(insert_shape))

        # Actually perform the slice.
        index = [slice(None) for _ in range(len(existing_mat.shape))]
        for i in range(num_sensor):
            index[axes[i]] = slice(self._x_data_idx[data_types[i]][0], self._x_data_idx[data_types[i]][-1] + 1)
        existing_mat[tuple(index)] = data_to_insert

    def pack_sample(self, X, U):
        """Packs sample data into Sample object."""
        assert X.shape[0] == self.T
        assert U.shape[0] == self.T
        assert X.shape[1] == self.dX
        assert U.shape[1] == self.dU

        sample = Sample(self)
        for sensor, idx in self._x_data_idx.items():
            sample.set(sensor, X[:, idx])
        for actuator, idx in self._u_data_idx.items():
            sample.set(actuator, U[:, idx])
        sample.set(ACTION, U)
        return sample
