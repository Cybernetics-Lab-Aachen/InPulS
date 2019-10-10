""" Default configuration for policy optimization. """
import os

# config options shared by both caffe and tf.
GENERIC_CONFIG = {
    # Initialization.
    'init_var': 0.1,  # Initial policy variance.
    'ent_reg': 0.0,  # Entropy regularizer.
    # Solver hyperparameters.
    'iterations': 5000,  # Number of iterations per inner iteration.
    'batch_size': 25,
    'lr': 0.001,  # Base learning rate (by default it's fixed).
    'lr_policy': 'fixed',  # Learning rate policy.
    'momentum': 0.9,  # Momentum.
    'weight_decay': 0.005,  # Weight decay.
    'solver_type': 'Adam',  # Solver type (e.g. 'SGD', 'Adam', etc.).
    # set gpu usage.
    'use_gpu': 0,  # Whether or not to use the GPU for caffe training.
    'gpu_id': 0,
    'random_seed': 1,
}

checkpoint_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'policy_opt/tf_checkpoint/policy_checkpoint.ckpt')
)
POLICY_OPT_TF = {
    # Other hyperparameters.
    'checkpoint_prefix': checkpoint_path
}

POLICY_OPT_TF.update(GENERIC_CONFIG)
