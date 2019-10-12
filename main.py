import logging
import imp
import os
from pathlib import Path
import sys
import argparse
import random
import numpy as np
from tqdm import trange

from gps.sample.sample_list import SampleList
from gps.algorithm.algorithm import Timer
from gps.visualization import visualize_trajectories

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Make tensorflow less chatty


class GPSMain(object):
    """ Main class to run algorithms and experiments. """

    def __init__(self, config):
        """
        Initialize GPSMain
        Args:
            config: Hyperparameters for experiment
        """
        self._hyperparams = config
        self._conditions = config['common']['conditions']
        if 'train_conditions' in config['common']:
            self._train_idx = config['common']['train_conditions']
            self._test_idx = config['common']['test_conditions']
        else:
            self._train_idx = range(self._conditions)
            config['common']['train_conditions'] = config['common']['conditions']
            self._hyperparams = config
            self._test_idx = self._train_idx

        self._data_files_dir = config['common']['data_files_dir']
        config['agent']['data_files_dir'] = self._data_files_dir
        config['algorithm']['data_files_dir'] = self._data_files_dir

        self.agent = config['agent']['type'](config['agent'])
        config['algorithm']['agent'] = self.agent

        self.algorithm = config['algorithm']['type'](config['algorithm'])
        self.algorithm._data_files_dir = self._data_files_dir
        if hasattr(self.algorithm, 'traj_opt'):
            self.algorithm.traj_opt._data_files_dir = self._data_files_dir
        if hasattr(self.algorithm, 'policy_opt'):
            self.algorithm.policy_opt._data_files_dir = self._data_files_dir

        self.algorithm.X_labels = sum(
            map(lambda sensor: [sensor] * self.agent.sensor_dims[sensor], self.agent.x_data_types), []
        )

    def run(self):
        """
        Run training by iteratively sampling and taking an iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns: None
        """
        if 'load_model' in self._hyperparams:
            self.iteration_count = self._hyperparams['load_model'][1]
            self.algorithm.policy_opt.iteration_count = self.iteration_count
            self.algorithm.policy_opt.restore_model(*self._hyperparams['load_model'])

            # Global policy static resets
            if self._hyperparams['num_pol_samples_static'] > 0:
                self.export_samples(
                    self._take_policy_samples(
                        N=self._hyperparams['num_pol_samples_static'], pol=self.algorithm.policy_opt.policy, rnd=False
                    ),
                    '_pol-static',
                    visualize=True
                )

            return

        for itr in range(self._hyperparams['iterations']):
            self.iteration_count = itr
            if hasattr(self.algorithm, 'traj_opt'):
                self.algorithm.traj_opt.iteration_count = itr
            if hasattr(self.algorithm, 'policy_opt'):
                self.algorithm.policy_opt.iteration_count = itr

            print("*** Iteration %02d ***" % itr)
            if itr == 0 and 'load_initial_samples' in self._hyperparams:
                # Load trajectory samples
                print('Loading initial samples ...')
                sample_files = self._hyperparams['load_initial_samples']
                traj_sample_lists = [[] for _ in range(self.algorithm.M)]
                for sample_file in sample_files:
                    data = np.load(sample_file)
                    X, U = data['X'], data['U']
                    assert X.shape[0] == self.algorithm.M
                    for m in range(self.algorithm.M):
                        for n in range(X.shape[1]):
                            traj_sample_lists[m].append(self.agent.pack_sample(X[m, n], U[m, n]))
                traj_sample_lists = [SampleList(traj_samples) for traj_samples in traj_sample_lists]
            else:
                # Take trajectory samples
                with Timer(self.algorithm.timers, 'sampling'):
                    for cond in self._train_idx:
                        for i in trange(self._hyperparams['num_samples'], desc='Taking samples'):
                            self._take_sample(itr, cond, i)
                traj_sample_lists = [
                    self.agent.get_samples(cond, -self._hyperparams['num_samples']) for cond in self._train_idx
                ]
            self.export_samples(traj_sample_lists, visualize=True)

            # Iteration
            with Timer(self.algorithm.timers, 'iteration'):
                self.algorithm.iteration(traj_sample_lists, itr)
            self.export_dynamics()
            self.export_controllers()
            self.export_times()
            if hasattr(self.algorithm, 'policy_opt') and hasattr(self.algorithm.policy_opt, 'store_model'):
                self.algorithm.policy_opt.store_model()

            # Sample learned policies for visualization

            # LQR policies static resets
            if self._hyperparams['num_lqr_samples_static'] > 0:
                self.export_samples(
                    self._take_policy_samples(N=self._hyperparams['num_lqr_samples_static'], pol=None, rnd=False),
                    '_lqr-static',
                    visualize=True
                )

            # LQR policies random resets
            if self._hyperparams['num_lqr_samples_random'] > 0:
                self.export_samples(
                    self._take_policy_samples(N=self._hyperparams['num_lqr_samples_random'], pol=None, rnd=True),
                    '_lqr-random',
                    visualize=True
                )

            # LQR policies state noise
            if self._hyperparams['num_lqr_samples_random'] > 0:
                self.export_samples(
                    self._take_policy_samples(
                        N=self._hyperparams['num_lqr_samples_random'], pol=None, rnd=False, randomize_initial_state=24
                    ),
                    '_lqr-static-randomized',
                    visualize=True
                )

            if hasattr(self.algorithm, 'policy_opt'):
                # Global policy static resets
                if self._hyperparams['num_pol_samples_static'] > 0:
                    self.export_samples(
                        self._take_policy_samples(
                            N=self._hyperparams['num_pol_samples_static'],
                            pol=self.algorithm.policy_opt.policy,
                            rnd=False
                        ),
                        '_pol-static',
                        visualize=True
                    )

                # Global policy random resets
                if self._hyperparams['num_pol_samples_random'] > 0:
                    self.export_samples(
                        self._take_policy_samples(
                            N=self._hyperparams['num_pol_samples_random'],
                            pol=self.algorithm.policy_opt.policy,
                            rnd=True
                        ),
                        '_pol-random',
                        visualize=True
                    )

                # Global policy state noise
                if self._hyperparams['num_pol_samples_random'] > 0:
                    self.export_samples(
                        self._take_policy_samples(
                            N=self._hyperparams['num_pol_samples_random'],
                            pol=self.algorithm.policy_opt.policy,
                            rnd=False,
                            randomize_initial_state=24
                        ),
                        '_pol-static-randomized',
                        visualize=True
                    )

            self.visualize_training_progress()

    def _take_sample(self, itr, cond, i):
        """
        Collect a sample from the agent.
        Args:
            itr: Iteration number.
            cond: Condition number.
            i: Sample number.
        Returns: None
        """
        if self.algorithm._hyperparams['sample_on_policy'] and self.algorithm.iteration_count > 0:
            pol = self.algorithm.policy_opt.policy
        else:
            pol = self.algorithm.cur[cond].traj_distr

        self.agent.sample(
            pol,
            cond,
            verbose=(i < self._hyperparams['verbose_trials']),
            noisy=True,
            use_TfController=True,
            reset_cond=None if self.agent._hyperparams['random_reset'] else cond
        )

    def _take_policy_samples(self, N, pol, rnd=False, randomize_initial_state=0):
        """
        Take samples from the policy to see how it's doing.
        Args:
            N  : number of policy samples to take per condition
            pol: Policy to sample. None for LQR policies.
        Returns: None
        """
        if pol is None:
            pol_samples = [[None] * N] * len(self._test_idx)
            for i, cond in enumerate(self._test_idx, 0):
                for n in trange(N, desc='Taking LQR-policy samples m=%d, cond=%s' % (cond, 'rnd' if rnd else cond)):
                    pol_samples[i][n] = self.agent.sample(
                        self.algorithm.cur[cond].traj_distr,
                        None,
                        verbose=None,
                        save=False,
                        noisy=False,
                        reset_cond=None if rnd else cond,
                        randomize_initial_state=randomize_initial_state,
                        record=False
                    )
            return [SampleList(samples) for samples in pol_samples]
        else:
            conds = self._test_idx if not rnd else [None]
            # stores where the policy has lead to
            pol_samples = [[None] * N] * len(conds)
            for i, cond in enumerate(conds):
                for n in trange(
                    N, desc='Taking %s policy samples cond=%s' % (type(pol).__name__, 'rnd' if rnd else cond)
                ):
                    pol_samples[i][n] = self.agent.sample(
                        pol,
                        None,
                        verbose=None,
                        save=False,
                        noisy=False,
                        reset_cond=cond,
                        randomize_initial_state=randomize_initial_state,
                        record=n < 0
                    )
            return [SampleList(samples) for samples in pol_samples]

    def export_samples(self, traj_sample_lists, sample_type='', visualize=False):
        """
        Exports trajectoy samples in a compressed numpy file.
        """
        M, N, T, dX, dU = len(traj_sample_lists), len(traj_sample_lists[0]), self.agent.T, self.agent.dX, self.agent.dU
        X = np.empty((M, N, T, dX))
        U = np.empty((M, N, T, dU))

        for m in range(M):
            sample_list = traj_sample_lists[m]
            for n in range(N):
                sample = sample_list[n]
                X[m, n] = sample.get_X()
                U[m, n] = sample.get_U()

        np.savez_compressed(
            self._data_files_dir + 'samples%s_%02d' % (sample_type, self.iteration_count),
            X=X,
            U=U,
        )

        if visualize:
            from gps.visualization.costs import visualize_costs

            X_labels = sum(map(lambda sensor: [sensor] * self.agent.sensor_dims[sensor], self.agent.x_data_types), [])
            U_labels = sum(map(lambda sensor: [sensor] * self.agent.sensor_dims[sensor], self.agent.u_data_types), [])

            for m in range(M):
                visualize_trajectories(
                    self._data_files_dir + 'samples%s_%02d-m%02d' % (sample_type, self.iteration_count, m),
                    X=X[m],
                    U=U[m],
                    X_labels=X_labels,
                    U_labels=U_labels
                )
                visualize_costs(
                    self._data_files_dir + 'samples%s_%02d-m%02d-costs' % (sample_type, self.iteration_count, m),
                    traj_sample_lists[m].get_samples(), self.algorithm.cost[m]
                )

    def export_dynamics(self):
        """
        Exports the local dynamics data in a compressed numpy file.
        """
        if self.algorithm.cur[0].traj_info.dynamics is None:
            return

        M, T, dX, dU = self.algorithm.M, self.agent.T, self.agent.dX, self.agent.dU
        Fm = np.empty((M, T - 1, dX, dX + dU))
        fv = np.empty((M, T - 1, dX))
        dyn_covar = np.empty((M, T - 1, dX, dX))

        for m in range(M):
            dynamics = self.algorithm.cur[m].traj_info.dynamics
            Fm[m] = dynamics.Fm[:-1]
            fv[m] = dynamics.fv[:-1]
            dyn_covar[m] = dynamics.dyn_covar[:-1]

        np.savez_compressed(
            self._data_files_dir + 'dyn_%02d' % self.iteration_count,
            Fm=Fm,
            fv=fv,
            dyn_covar=dyn_covar,
        )

    def export_controllers(self):
        """
        Exports the local controller data in a compressed numpy file.
        """
        if self.algorithm.cur[0].traj_distr is None:
            return

        M, T, dX, dU = self.algorithm.M, self.agent.T, self.agent.dX, self.agent.dU
        K = np.empty((M, T - 1, dU, dX))
        k = np.empty((M, T - 1, dU))
        prc = np.empty((M, T - 1, dU, dU))

        traj_mu = np.empty((M, T, dX + dU))
        traj_sigma = np.empty((M, T, dX + dU, dX + dU))

        for m in range(M):
            traj = self.algorithm.cur[m].traj_distr
            K[m] = traj.K[:-1]
            k[m] = traj.k[:-1]
            prc[m] = traj.inv_pol_covar[:-1]
            traj_mu[m] = self.algorithm.new_mu[m]
            traj_sigma[m] = self.algorithm.new_sigma[m]

        np.savez_compressed(
            self._data_files_dir + 'ctr_%02d' % self.iteration_count,
            K=K,
            k=k,
            prc=prc,
            traj_mu=traj_mu,
            traj_sigma=traj_sigma,
        )

    def export_times(self):
        """
        Exports timer values into a csv file by appending a line for each iteration.
        """
        header = ','.join(self.algorithm.timers.keys()) if self.iteration_count == 0 else ''
        with open(self._data_files_dir + 'timers.csv', 'ab') as out_file:
            np.savetxt(out_file, np.asarray([np.asarray([f for f in self.algorithm.timers.values()])]), header=header)

    def visualize_training_progress(self):
        if 'traing_progress_metric' in self._hyperparams:
            from gps.visualization.training import visualize_training

            visualize_training(
                self._data_files_dir + 'progress',
                [
                    {
                        'experiment': self._data_files_dir,
                        'label': sample_type,
                        'sample_type': sample_type,
                    } for key, sample_type in [
                        (None, 'samples'),
                        ('num_lqr_samples_static', 'samples_lqr-static'),
                        ('num_lqr_samples_random', 'samples_lqr-random'),
                        ('num_pol_samples_static', 'samples_pol-static'),
                        ('num_pol_samples_random', 'samples_pol-random'),
                    ] if key is None or self._hyperparams[key] > 0
                ],
                metric=self._hyperparams['traing_progress_metric'],
                target=0.01,
            )


if __name__ == "__main__":
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Run the Guided Policy Search algorithm.')
    parser.add_argument('experiment', type=str, help='experiment name')
    args = parser.parse_args()

    exp_name = args.experiment

    hyperparams_file = Path('experiments/') / exp_name / 'hyperparams.py'

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARN)

    if not hyperparams_file.is_file():
        sys.exit("Experiment '%s' does not exist.\nDid you create '%s'?" % (exp_name, hyperparams_file))

    hyperparams = imp.load_source('hyperparams', str(hyperparams_file))

    seed = hyperparams.config.get('random_seed', 0)
    random.seed(seed)
    np.random.seed(seed)
    gps = GPSMain(hyperparams.config)
    gps.run()
