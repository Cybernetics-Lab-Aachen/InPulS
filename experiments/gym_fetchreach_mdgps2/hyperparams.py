"""Hyperparameters for FetchReach task using new MDGPS implemntation."""

from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

from __main__ import __file__ as main_filepath
from gps.agent.openai_gym import AgentOpenAIGym
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.cost import CostAction, CostState, CostSum
from gps.algorithm.dynamics import DynamicsLRPrior, DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.gps import GPS_Policy
from gps.agent.openai_gym.init_policy import init_gym_pol
from gps.proto.gps_pb2 import END_EFFECTOR_POINTS, ACTION
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM

SENSOR_DIMS = {
    'observation': 10,
    END_EFFECTOR_POINTS: 3,
    ACTION: 4,
}

EXP_DIR = str(Path(__file__).parent).replace('\\', '/') + '/'

common = {
    'data_files_dir': EXP_DIR + 'data_files/',
    'conditions': 4,
    # 'train_conditions': [0],
    # 'test_conditions': [0, 1, 2, 3],
}

scaler = StandardScaler()
scaler.mean_ = [
    1.33554446e+00, 7.46866838e-01, 5.25169272e-01, 3.78054915e-06, 1.55438229e-06, -2.66267931e-04, -4.78114576e-05,
    -5.77911271e-05, 4.03657748e-04, 4.67200411e-04, 6.71023554e-03, -1.16670921e-02, -6.79674174e-03
]
scaler.scale_ = [
    1.07650035e-01, 1.29246324e-01, 9.07454956e-02, 2.64638440e-05, 1.08806760e-05, 7.48034005e-03, 7.69515138e-03,
    7.54787489e-03, 1.37551241e-03, 1.52642517e-03, 7.57966860e-02, 1.03377066e-01, 6.24320497e-02
]

agent = {
    'type': AgentOpenAIGym,
    'render': False,
    'T': 20,
    'random_reset': False,
    'x0': [0, 1, 2, 3, 4, 5, 6, 7],  # Random seeds for each initial condition
    'dt': 1.0 / 25,
    'env': 'FetchReach-v1',
    'sensor_dims': SENSOR_DIMS,
    'target_state': scaler.transform([np.zeros(13)])[0, -3:],
    'conditions': common['conditions'],
    'state_include': ['observation', END_EFFECTOR_POINTS],
    'obs_include': ['observation', END_EFFECTOR_POINTS],
    'actions_include': [ACTION],
    'scaler': scaler,
}

algorithm = {
    'type': AlgorithmMDGPS,
    'conditions': common['conditions'],
    'iterations': 20,
    'kl_step': 1.0,
    'min_step_mult': 0.5,
    'max_step_mult': 3.0,
    'policy_sample_mode': 'replace',
    'sample_on_policy': False,
}

algorithm['init_traj_distr'] = {
    'type': init_gym_pol,
    'init_var_scale': 1.0,
    'env': agent['env'],
    'T': agent['T'],
}

action_cost = {
    'type': CostAction,
    'wu': np.ones(SENSOR_DIMS[ACTION]),
    'name': 'Action',
    'target_state': np.zeros(SENSOR_DIMS[ACTION]),
}

state_cost = {
    'type': CostState,
    'data_types': {
        END_EFFECTOR_POINTS: {
            'wp': np.ones(3),  # Target size
            'target_state': agent["target_state"],
        },
    },
    'name': 'EE dist',
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [action_cost, state_cost],
    'weights': [1E-3, 1.0],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior':
        {
            'type': DynamicsPriorGMM,
            'max_clusters': 8,
            'min_samples_per_cluster': 40,
            'max_samples': 40,
            'strength': 1,
        },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {
    'type': GPS_Policy,
    'random_seed': 1,
    'init_var': 0.1,
    'ent_reg': 0.0,
    'epochs': 100,
    'batch_size': (agent['T'] - 1) * min(2, common['conditions']),  # batch size must be divisor of (T-1) * M * N
    'weight_decay': 0.005,
    'N_hidden': 80,
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 40,
}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 5,
    'num_lqr_samples_static': 1,
    'num_lqr_samples_random': 5,
    'num_pol_samples_static': 1,
    'num_pol_samples_random': 20,
    'common': common,
    'agent': agent,
    'algorithm': algorithm,
    'random_seed': 0,
}

param_str = 'fetchreach_gps2'
baseline = True
param_str += '-random' if agent['random_reset'] else '-static'
param_str += '-M%d' % config['common']['conditions']
param_str += '-%ds' % config['num_samples']
param_str += '-T%d' % agent['T']
param_str += '-K%d' % algorithm['dynamics']['prior']['max_clusters']
param_str += '-h%r' % algorithm['policy_opt']['N_hidden']
common['data_files_dir'] += '%s_%d/' % (param_str, config['random_seed'])

# Only make changes to filesystem if loaded by training process
if Path(main_filepath) == Path(__file__).parents[2] / 'main.py':
    from shutil import copy2

    # Make expirement folder and copy hyperparams
    Path(common['data_files_dir']).mkdir(parents=True, exist_ok=False)
    copy2(EXP_DIR + 'hyperparams.py', common['data_files_dir'])
