""" Hyperparameters for JACO trajectory optimization experiment. """
from __future__ import division

from datetime import datetime
import os.path
import time

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.ros_jaco.agent_ros_jaco import AgentROSJACO
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.gui.target_setup_gui import load_pose_from_npz
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION, \
        TRIAL_ARM, AUXILIARY_ARM, JOINT_SPACE, RGB_IMAGE, RGB_IMAGE_SIZE
from gps.utility.general_utils import get_ee_points




EE_POINTS = np.array([[0.04, -0.10, -0.19], [-0.04, -0.10, -0.19], [0.0, -0.08, -0.12]])

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
IMAGE_CHANNELS = 3

SENSOR_DIMS = {
    JOINT_ANGLES: 6,
    JOINT_VELOCITIES: 6,
    END_EFFECTOR_POINTS: 3 * EE_POINTS.shape[0],
    END_EFFECTOR_POINT_VELOCITIES: 3 * EE_POINTS.shape[0],
    RGB_IMAGE: IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS,
    ACTION: 6,
}

PR2_GAINS = np.array([7.09, 2.3, 1.5, 1.2, 1.8, 0.1])

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/jaco_example/'

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'cost_log_dir': EXP_DIR + 'cost_log/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    #'train_conditions': [0, 1],
    #'test_conditions': [1,2],
    'conditions': 2,
    'experiment_ID': '1' + time.ctime(),
}

# Set up each condition.
x0s = []
x_tgts = []
ee_tgts = []
reset_conditions = []
for i in xrange(common['conditions']):

    ja_x0_, ee_pos_x0, ee_rot_x0 = load_pose_from_npz(
        common['target_filename'], 'trial_arm', str(i), 'initial'
    )
    ja_tgt, ee_pos_tgt, ee_rot_tgt = load_pose_from_npz(
        common['target_filename'], 'trial_arm', str(i), 'target'
    )
    x_tgt = np.zeros(12)
    jv_tgt = np.zeros(6)
    x_tgt[:6] = ja_tgt
    x_tgt[6:12] = jv_tgt

    ja_x0 = ja_x0_[:6]
    x0 = np.zeros(12)
    x0[:6] = ja_x0
    #x0[12:(12+3*EE_POINTS.shape[0])] = np.ndarray.flatten(
    #    get_ee_points(EE_POINTS, ee_pos_x0, ee_rot_x0).T
    #)

    ee_tgt = np.ndarray.flatten(
        get_ee_points(EE_POINTS, ee_pos_tgt, ee_rot_tgt).T
    )

    reset_condition = {
        TRIAL_ARM: {
            'mode': JOINT_SPACE,
            'data': x0[0:6],
        },
    }

    x_tgts.append(x_tgt)
    x0s.append(x0)
    ee_tgts.append(ee_tgt)
    reset_conditions.append(reset_condition)

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentROSJACO,
    'dt': 0.05,
    'conditions': common['conditions'],
    'T': 80,
    'x0': x0s,
    'ee_points': EE_POINTS,
    'dee_tgt': 3 * EE_POINTS.shape[0],
    'exp_x_tgts': x_tgts,
    'dtgtX': 6,
    'include_tgt': False,
    'ee_points_tgt': ee_tgts,
    'reset_conditions': reset_conditions,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES],
    'end_effector_points': EE_POINTS,
    'obs_include': [],
    'rgb_topic': '/usb_cam/image_raw',
    'rgb_shape': [IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS],
}


algorithm = {
    'type': AlgorithmTrajOpt,
    #'train_conditions': common['train_conditions'],
    #'test_conditions': common['test_conditions'],
    'conditions': common['conditions'],
    'exp_x_tgts': x_tgts,
    'include_tgt': agent['include_tgt'],
    'dtgtX': agent['dtgtX'],
    'dee_tgt': agent['dee_tgt'],
    'ee_points_tgt': agent['ee_points_tgt'],
    'iterations': 25,
}


algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains':  1.0 / PR2_GAINS,
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 1.0,
    'stiffness': 1.0,
    'stiffness_vel': 0.25,
    'final_weight': 50.0,
    'dt': agent['dt'],
    'T': agent['T'],
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

torque_cost = {
    'type': CostAction,
    'wu': 5e-3 / PR2_GAINS,
}

'''
fk_cost1 = {
   'type': CostFK,
    # Target end effector is subtracted out of EE_POINTS in ROS so goal
    # is 0.
    'target_end_effector': np.zeros(3 * EE_POINTS.shape[0]),
    'wp': np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-6,
    'experiment_ID': common['experiment_ID'],
    'dir':common['cost_log_dir'],
}

fk_cost2 = {
    'type': CostFK,
    'ramp_option': RAMP_FINAL_ONLY,
    'target_end_effector': fk_cost1['target_end_effector'],
    'wp': fk_cost1['wp'],
    'l1': 1.0,
    'l2': 15.0,
    'alpha': 1e-6,
    'wp_final_multiplier': 25.0,
    'experiment_ID': common['experiment_ID'],
    'dir':common['cost_log_dir'],
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [torque_cost, fk_cost1, fk_cost2],
    'weights': [1.0, 1.0, 1.0],
}
'''

state_cost = {
    'type': CostState,
    'data_types' : {
        JOINT_ANGLES: {
            'wp': np.array([1, 1, 1, 1, 1, 1]),
            'target_state': agent["exp_x_tgts"][0][:6],
        },
    },
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [torque_cost, state_cost],
    'weights': [1.0, 1.0],
}
algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 40,
        'min_samples_per_cluster': 40,
        'max_samples': 30,
    },
}

config = {
    'iterations': algorithm['iterations'],
    'common': common,
    'verbose_trials': 10,
    'verbose_policy_trials': 1,
    'min_iteration_for_testing' : 15,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
    'num_samples': 5,
    'experiment_ID': common['experiment_ID'],
    'dir':common['cost_log_dir'],
}

common['info'] = (
    'exp_name: ' + str(common['experiment_name'])              + '\n'
    'alg_type: ' + str(algorithm['type'].__name__)             + '\n'
    'alg_dyn:  ' + str(algorithm['dynamics']['type'].__name__) + '\n'
    'alg_cost: ' + str(algorithm['cost']['type'].__name__)     + '\n'
    'iterations: ' + str(config['iterations'])                   + '\n'
    'conditions: ' + str(algorithm['conditions'])                + '\n'
    'samples:    ' + str(config['num_samples'])                  + '\n'
)
