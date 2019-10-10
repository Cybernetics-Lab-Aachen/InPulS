""" Default configuration and hyperparameter values for policies. """

# PolicyPrior
POLICY_PRIOR = {
    'strength': 1e-4,
}

# PolicyPriorGMM
POLICY_PRIOR_GMM = {
    'min_samples_per_cluster': 20,
    'max_clusters': 50,
    'max_samples': 20,
    'strength': 1.0,
}
