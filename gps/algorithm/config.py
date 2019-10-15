"""Default configuration and hyperparameter values for algorithms."""

# Algorithm
ALG = {
    'min_eta': 1e-5,  # Minimum initial lagrange multiplier in DGD for
    # trajectory optimization.
    'kl_step': 0.2,
    'min_step_mult': 0.01,
    'max_step_mult': 10.0,
    # Trajectory settings.
    'initial_state_var': 1e-6,
    'init_traj_distr': None,  # A list of initial LinearGaussianPolicy
    # objects for each condition.
    # Trajectory optimization.
    'traj_opt': None,
    # Dynamics hyperaparams.
    'dynamics': None,
    # Costs.
    'cost': None,  # A list of Cost objects for each condition.
    # Whether or not to sample with neural net policy (only for badmm/mdgps).
    'sample_on_policy': False,
}

# AlgorithmMD
ALG_MDGPS = {
    'policy_sample_mode': 'add',
    # Whether to use 'laplace' or 'mc' cost in step adjusment
    'step_rule': 'laplace',
}
