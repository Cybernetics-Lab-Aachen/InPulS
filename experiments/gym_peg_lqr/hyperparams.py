""" Hyperparameters for Box2d Point Mass."""
from __future__ import division

import os.path
from os import mkdir
from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler

from gps import __file__ as gps_filepath
from gps.agent.openai_gym.agent_openai_gym import AgentOpenAIGym
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.agent.openai_gym.init_policy import init_gym_pol
from gps.gui.config import generate_experiment_info
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, ACTION
import gps.envs

SENSOR_DIMS = {
    JOINT_ANGLES: 7,
    JOINT_VELOCITIES: 7,
    END_EFFECTOR_POINTS: 6,
    'diff': 6,
    ACTION: 7,
}

PR2_GAINS = np.array([3.09, 1.08, 0.393, 0.674, 0.111, 0.152, 0.098])

BASE_DIR = '/'.join(str.split(gps_filepath.replace('\\', '/'), '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/gym_peg_lqr/'


common = {
    'experiment_name': 'gym_peg_lqr' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
    # 'train_conditions': [0],
    # 'test_conditions': [0, 1, 2, 3],
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

scaler = StandardScaler()
scaler.mean_ = []
scaler.scale_ = []


def additional_sensors(sim, sample, t):
    from gps.proto.gps_pb2 import END_EFFECTOR_POINT_JACOBIANS
    jac = np.empty((6, 7))
    jac[:3] = sim.data.get_site_jacp('leg_bottom').reshape((3, -1))
    jac[3:] = sim.data.get_site_jacp('leg_top').reshape((3, -1))
    sample.set(END_EFFECTOR_POINT_JACOBIANS, jac, t=t)


agent = {
    'type': AgentOpenAIGym,
    'render': True,
    'T': 150,
    'random_reset': False,
    'x0': [0, 1, 2, 3, 4, 5, 6, 7],  # Random seeds for each initial condition
    'dt': 1.0/25,
    'env': 'PegInsertion-v0',
    'sensor_dims': SENSOR_DIMS,
    'target_state': np.zeros(3),  # Target np.zeros(3), 
    'conditions': common['conditions'],
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, 'diff'],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, 'diff'],
#    'scaler': scaler,
    'additional_sensors': additional_sensors,
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
    'iterations': 100,
    #'tac_policy': {
    #    'history': 10,
    #},
}

algorithm['init_traj_distr'] = {
    'type': init_gym_pol,
    'env': agent['env'],
    'T': agent['T'],
}

torque_cost = {
    'type': CostAction,
    'wu': 1.0 / PR2_GAINS,
}

fk_cost = {
    'type': CostFK,
    'target_end_effector': np.array([0.0, 0.3, -0.5, 0.0, 0.3, -0.2]),
    'wp': np.array([1, 1, 1, 1, 1, 1]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
}

state_cost = {
    'type': CostState,
    'data_types': {
        'diff': {
            'wp': np.ones(6),  # Target size
            'target_state': np.zeros(6),
        },
    },
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [torque_cost, fk_cost],
    'weights': [1e-3, 1.0],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
        'strength': 1,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 20,
    'num_pol_samples': 5,
    'save_samples': False,
    'verbose_trials': 0,
    'common': common,
    'agent': agent,
    'gui_on': False,
    'algorithm': algorithm,
    'random_seed': 0,
}

common['info'] = generate_experiment_info(config)

param_str = 'fetchreach_lqr'
baseline = True
param_str += '-random' if agent['random_reset'] else '-static'
param_str += '-M%d' % config['common']['conditions']
param_str += '-%ds' % config['num_samples']
param_str += '-T%d' % agent['T']
param_str += '-tac_pol' if 'tac_policy' in algorithm else '-lqr_pol'
common['data_files_dir'] += '%s_%d/' % (param_str, config['random_seed'])
mkdir(common['data_files_dir'])
