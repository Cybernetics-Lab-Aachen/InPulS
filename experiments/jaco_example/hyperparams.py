""" Hyperparameters for JACO trajectory optimization experiment. """
from __future__ import division

from datetime import datetime
import os.path
import time

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.ros_jaco.agent_ros_jaco import AgentROSJACO
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.gui.target_setup_gui import load_pose_from_npz
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION, \
        TRIAL_ARM, AUXILIARY_ARM, JOINT_SPACE
from gps.utility.general_utils import get_ee_points


#EE_POINTS = np.array([[0.02, -0.02, 0.035], [0.02, 0.02, 0.035], [-0.02, 0.0, 0.005]])
EE_POINTS = np.array([[-0.04, -0.04, 0.182], [-0.04, 0.04, 0.182], [0.04, 0.0, 0.142]])

SENSOR_DIMS = {
    JOINT_ANGLES: 6,
    JOINT_VELOCITIES: 6,
    END_EFFECTOR_POINTS: 3 * EE_POINTS.shape[0],
    END_EFFECTOR_POINT_VELOCITIES: 3 * EE_POINTS.shape[0],
    ACTION: 6,
}

#PR2_GAINS = np.array([3.09, 1.08, 0.393, 0.674, 0.111, 0.152, 0.098])

PR2_GAINS = np.array([7.09, 2.3, 1.5, 1.2, 1.8, 1.3])

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/jaco_example/'

x0s = []
ee_tgts = []
reset_conditions = []

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'cost_log_dir': EXP_DIR + 'cost_log/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
    'experiment_ID': '1' + time.ctime(),
}

# TODO(chelsea/zoe) : Move this code to a utility function
# Set up each condition.

for i in xrange(common['conditions']):

    ja_x0_, ee_pos_x0, ee_rot_x0 = load_pose_from_npz(
        common['target_filename'], 'trial_arm', str(i), 'initial'
    )
    _, ee_pos_tgt, ee_rot_tgt = load_pose_from_npz(
        common['target_filename'], 'trial_arm', str(i), 'target'
    )
    ja_x0 = ja_x0_[:6]
    x0 = np.zeros(30)
    x0[:6] = ja_x0
    x0[12:(12+3*EE_POINTS.shape[0])] = np.ndarray.flatten(
        get_ee_points(EE_POINTS, ee_pos_x0, ee_rot_x0).T
    )
    print "ja_x0"
    print ja_x0
    print "ee_pos_x0"
    print ee_pos_x0
    print "ee_rot_x0"
    print ee_rot_x0
    print "x0"
    print x0

    ee_tgt = np.ndarray.flatten(
        get_ee_points(EE_POINTS, ee_pos_tgt, ee_rot_tgt).T
    )

    pi = 3.14159265359
    #reset_jointpos = [0,0,0,0,0,0,0]
    reset_jointpos = [pi/2,pi+pi/8,pi/2,pi+pi/8,pi/8,pi+pi/4]
    reset_condition = {
        TRIAL_ARM: {
            'mode': JOINT_SPACE,
            'data': x0[0:6],
        },
    }

    x0s.append(x0)
    ee_tgts.append(ee_tgt)
    reset_conditions.append(reset_condition)
    #raw_input(np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]))

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentROSJACO,
    'dt': 0.05,
    'conditions': common['conditions'],
    'T': 80,
    'x0': x0s,
    'ee_points_tgt': ee_tgts,
    'reset_conditions': reset_conditions,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
    'end_effector_points': EE_POINTS,
    'obs_include': [],
}

'''
agent = {
    'type': AgentROSIIWA,
    'x0': np.concatenate([np.array([0.1, 0.1, -1.54, -1.7, 1.54, -0.2, 0]),
                          np.zeros(7)]),
    'dt': 0.05,
    'substeps': 5,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([1]),
    'pos_body_offset': [np.array([0, 0.2, 0]), np.array([0, 0.1, 0]),
                        np.array([0, -0.1, 0]), np.array([0, -0.2, 0])],
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [],
    'camera_pos': np.array([0., 0., 2., 0., 0.2, 0.5]),
}
'''

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
    'iterations': 25,
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains': 1.0 / np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
    #'init_gains':  1.0 / np.array([7.09, 3.1, 3.1, 1.2, 1.8, 1.3]),
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 1.0,
    'stiffness': 1.0,
    'stiffness_vel': 0.25,
    'final_weight': 50.0,
    'dt': agent['dt'],
    'T': agent['T'],
}

'''
torque_cost = {
    'type': CostAction,
    'wu': 5e-5 / PR2_GAINS,
}

fk_cost1 = {
    'type': CostFK,
    # Target end effector is subtracted out of EE_POINTS in ROS so goal
    # is 0.
    'target_end_effector': np.zeros(3 * EE_POINTS.shape[0]),
    'wp': np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
    #'evalnorm': evallogl2term,
    #'ramp_option': RAMP_LINEAR,
}

fk_cost2 = {
    'type': CostFK,
    'target_end_effector': np.zeros(3 * EE_POINTS.shape[0]),
    'wp': np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]),
    'l1': 1.0,
    'l2': 2.0,
    'wp_final_multiplier': 10.0,  # Weight multiplier on final timestep.
    'ramp_option': RAMP_FINAL_ONLY,
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [torque_cost, fk_cost1, fk_cost2],
    'weights': [1.0, 1.0, 1.0],
}

'''

torque_cost = {
    'type': CostAction,
    'wu': 5e-4 / PR2_GAINS,
}

fk_cost1 = {
   'type': CostFK,
    # Target end effector is subtracted out of EE_POINTS in ROS so goal
    # is 0.
    'target_end_effector': np.zeros(3 * EE_POINTS.shape[0]),
    'wp': np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]),
#    'l1': 0.005,
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-6,
    'experiment_ID': common['experiment_ID'],
    'dir':common['cost_log_dir'],
#    'alpha': 1e-10,
    #'evalnorm': evallogl2term,
    #'ramp_option': RAMP_LINEAR,
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

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {}

config = {
    'iterations': algorithm['iterations'],
    'common': common,
    'verbose_trials': 10,
    #'verbose_policy_trials': 2,
    'min_iteration_for_testing' : 15,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
    'num_samples': 6,
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
