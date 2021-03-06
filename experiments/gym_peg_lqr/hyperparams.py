"""Hyperparameters for PR2 peg in hole task using LQR."""

from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

from __main__ import __file__ as main_filepath
from gps.agent.openai_gym import AgentOpenAIGym
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost import CostAction, CostFK, CostSum
from gps.algorithm.dynamics import DynamicsLRPrior, DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.agent.openai_gym.init_policy import init_gym_pol
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, ACTION
import gps.envs

SENSOR_DIMS = {
    JOINT_ANGLES: 7,
    JOINT_VELOCITIES: 7,
    END_EFFECTOR_POINTS: 6,
    'diff': 6,
    ACTION: 7,
}

EXP_DIR = str(Path(__file__).parent).replace('\\', '/') + '/'

common = {
    'data_files_dir': EXP_DIR + 'data_files/',
    'conditions': 4,
    # 'train_conditions': [0],
    # 'test_conditions': [0, 1, 2, 3],
}


def additional_sensors(sim, sample, t):
    """Virtual sensors supplying the leg positions."""
    from gps.proto.gps_pb2 import END_EFFECTOR_POINT_JACOBIANS
    jac = np.empty((6, 7))
    jac[:3] = sim.data.get_site_jacp('leg_bottom').reshape((3, -1))
    jac[3:] = sim.data.get_site_jacp('leg_top').reshape((3, -1))
    sample.set(END_EFFECTOR_POINT_JACOBIANS, jac, t=t)


scaler = StandardScaler()
scaler.mean_ = [
    6.269170526989192860e-01, 4.821029797409001616e-01, -1.544978642130366175e+00, -1.176340602904814014e+00,
    5.961119337564275977e-03, -1.430743556156053087e+00, 9.207545508118145094e-02, -4.912633791830883865e-04,
    4.920776339259459904e-03, -5.227700198026216660e-02, -1.211733843546966144e-01, 8.114241896172087742e-02,
    -1.262334659084702604e-01, 3.840017703328454934e-02, 2.194164582447949707e-01, 3.117020534855194369e-01,
    -2.769765942447273699e-01, 2.246371605649680747e-01, 2.734939294257240916e-01, -1.879409311259450099e-01,
    -2.194164582447949707e-01, -1.170205348552046803e-02, -2.230234057552664406e-01, -2.246371605649680747e-01,
    2.650607057427677160e-02, -1.205906887405655374e-02
]
scaler.scale_ = [
    1.175474491141771521e-01, 1.418192829274926015e-01, 7.290010554458773440e-01, 4.690964360306257297e-01,
    1.803339760782196821e+00, 5.855882638525999884e-01, 2.440150963082939661e+00, 7.986024203846138481e-02,
    8.789967075190831258e-02, 7.810188187563219531e-01, 4.007383765581046253e-01, 1.621819389085616736e+00,
    1.115074309447028122e+00, 1.874669395375669012e+00, 2.366287313117650670e-01, 1.671869278950406934e-01,
    1.699603332509740106e-01, 2.305784025129126447e-01, 1.472527037676900352e-01, 1.478939140218039627e-01,
    2.366287313117650670e-01, 1.671869278950406934e-01, 1.699603332509738718e-01, 2.305784025129126447e-01,
    1.472527037676900352e-01, 1.478939140218039905e-01
]

agent = {
    'type': AgentOpenAIGym,
    'render': False,
    'T': 100,
    'random_reset': False,
    'x0': [131, 327, 356, 491, 529, 853, 921, 937],  # Random seeds for each initial condition, 1
    'dt': 1.0 / 25,
    'env': 'PegInsertion-v0',
    'sensor_dims': SENSOR_DIMS,
    'conditions': common['conditions'],
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, 'diff'],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, 'diff'],
    'actions_include': [ACTION],
    # 'scaler': scaler,
    'additional_sensors': additional_sensors,
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
    'iterations': 50,
}

algorithm['init_traj_distr'] = {
    'type': init_gym_pol,
    'init_var_scale': 0.1,
    'env': agent['env'],
    'T': agent['T'],
}

action_cost = {
    'type': CostAction,
    'wu': np.ones(SENSOR_DIMS[ACTION]),
    'name': 'Action',
    'target_state': np.zeros(SENSOR_DIMS[ACTION]),
}

fk_cost = {
    'type': CostFK,
    'target_end_effector': np.array([0.0, 0.3, -0.5, 0.0, 0.3, -0.2]),
    'wp': np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
    'name': 'FK',
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [action_cost, fk_cost],
    'weights': [1e-3, 1.0],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior':
        {
            'type': DynamicsPriorGMM,
            'max_clusters': 20,
            'min_samples_per_cluster': 40,
            'max_samples': 5 * common['conditions'],
            'strength': 1,
        },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 5,
    'num_lqr_samples_static': 1,
    'num_lqr_samples_random': 5,
    'num_pol_samples_static': 1,
    'num_pol_samples_random': 20,
    'save_samples': False,
    'common': common,
    'agent': agent,
    'algorithm': algorithm,
    'random_seed': 0,
}

param_str = 'peg_lqr'
param_str += '-random' if agent['random_reset'] else '-static'
param_str += '-M%d' % config['common']['conditions']
param_str += '-%ds' % config['num_samples']
param_str += '-T%d' % agent['T']
param_str += '-K%d' % algorithm['dynamics']['prior']['max_clusters']
common['data_files_dir'] += '%s_%d/' % (param_str, config['random_seed'])

# Only make changes to filesystem if loaded by training process
if Path(main_filepath) == Path(__file__).parents[2] / 'main.py':
    from shutil import copy2

    # Make expirement folder and copy hyperparams
    Path(common['data_files_dir']).mkdir(parents=True, exist_ok=False)
    copy2(EXP_DIR + 'hyperparams.py', common['data_files_dir'])
