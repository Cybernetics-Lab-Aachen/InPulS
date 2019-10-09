from gps_main import GPSMain

import logging
import imp
import os
import os.path
import sys
import argparse
import threading
import time

# Add gps/python to path so that imports work. Replace backslashes for windows compability
sys.path.append('/'.join(str.split(__file__.replace('\\', '/'), '/')[:-2]))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Run the Guided Policy Search algorithm.')
    parser.add_argument('experiment', type=str, help='experiment name')
    parser.add_argument('-tc', '--testcond', nargs='+', type=int, help='test policy [iteration, start condition]')
    parser.add_argument('-t', '--targetsetup', action='store_true', help='run target setup')
    parser.add_argument('-r', '--resume', metavar='N', type=int, help='resume training from iter N')
    parser.add_argument('-p', '--policy', metavar='N', type=int, help='take N policy samples (for BADMM/MDGPS only)')
    args = parser.parse_args()

    exp_name = args.experiment
    resume_training_itr = args.resume
    test_policy_N = args.policy

    from gps import __file__ as gps_filepath
    gps_filepath = os.path.abspath(gps_filepath).replace('\\', '/')  # Replace backslashes for windows compability
    gps_dir = '/'.join(str.split(gps_filepath, '/')[:-3]) + '/'
    exp_dir = gps_dir + 'experiments/' + exp_name + '/'
    hyperparams_file = exp_dir + 'hyperparams.py'

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARN)

    if not os.path.exists(hyperparams_file):
        sys.exit("Experiment '%s' does not exist.\nDid you create '%s'?" % (exp_name, hyperparams_file))

    hyperparams = imp.load_source('hyperparams', hyperparams_file)

    if args.testgcm:
        import random
        import copy
        import numpy as np
        import matplotlib.pyplot as plt
        from gps.algorithm.gcm.gcm_controller import GCMController
        import gps.algorithm.gcm.gcm_utils as gcm_utils
        data_files_dir = exp_dir + 'data_files/'
        seed = hyperparams.config.get('random_seed', 0)
        random.seed(seed)
        np.random.seed(seed)

        gps = GPSMain(hyperparams.config)

        model_files_dir = data_files_dir + ('itr_%02d/' % args.testgcm)

        gcm_policy = GCMController(hyperparams.config, model_files_dir)
        gcm_policy.reset_ctr_context()

        #sample based on init configuration in hyperparams
        gcm_sample_lists = gps._take_policy_samples(1, gcm_policy)
        for i in range(len(gcm_sample_lists)):
            gcm_sample_lists[i].get_X()  # Fill in data for pickling
        gps.data_logger.pickle(
            data_files_dir + ('pol_gcm_sample_itr_%02d.pkl' % args.testgcm), copy.copy(gcm_sample_lists)
        )

        # sample latent space to verify controller models
        # plot only in case of a two dimensional latent space
        if gcm_policy.FLAGS.lat_x_dim == 2:
            Ks = []
            ks = []
            particle_number = 10
            for i in range(particle_number):
                for j in range(particle_number):
                    idx_i = i - particle_number / 2
                    idx_j = j - particle_number / 2
                    pn_i = idx_i / particle_number
                    pn_j = idx_j / particle_number
                    sample_z = np.array([pn_i, pn_j])
                    cur_ctr, cur_x = gcm_policy.sample_state_controller(np.reshape(sample_z, [1, -1]))
                    K, k = gcm_utils.deserialize_controller(cur_ctr, 6, 12)
                    Ks.append(K)
                    ks.append(k)
            gcm_policy.plot_controller(Ks, ks)

        sys.exit()

    if args.testgcmcond:
        import random
        import copy
        import numpy as np
        import matplotlib.pyplot as plt
        from gps.algorithm.gcm.gcm_controller import GCMController
        from gps.algorithm.algorithm_ggcs import AlgorithmGGCS
        import gps.algorithm.gcm.gcm_utils as gcm_utils

        labels = args.testgcmcond
        reset_cond = labels[1]

        print("reset to condition ", reset_cond)
        data_files_dir = exp_dir + 'data_files/'
        seed = hyperparams.config.get('random_seed', 0)
        random.seed(seed)
        np.random.seed(seed)
        gps = GPSMain(hyperparams.config, no_algorithm=False)
        model_files_dir = data_files_dir + ('itr_%02d/' % labels[0])
        if (type(gps.algorithm) == AlgorithmGGCS):
            gcm = gps.algorithm.policy_opt.gcm
            gcm_policy = GCMController(hyperparams.config, model_files_dir, restore=True, gcm=gcm)
        else:
            gcm_policy = GCMController(hyperparams.config, model_files_dir)
            gcm_policy.reset_ctr_context()

        # sample based on init configuration in hyperparams
        gcm_sample_lists = gps._take_policy_samples(1, gcm_policy, reset_cond)
        for i in range(len(gcm_sample_lists)):
            gcm_sample_lists[i].get_X()  # Fill in data for pickling
        gps.data_logger.pickle(
            data_files_dir + ('pol_gcm_sample_itr_%02d.pkl' % labels[0]), copy.copy(gcm_sample_lists)
        )
        sys.exit()

    if args.testcond:
        import random
        import numpy as np
        import matplotlib.pyplot as plt

        labels = args.testcond

        seed = hyperparams.config.get('random_seed', 0)
        random.seed(seed)
        np.random.seed(seed)

        gps = GPSMain(hyperparams.config)
        if hyperparams.config['gui_on']:
            test_policy = threading.Thread(target=lambda: gps.test_policy(itr=labels[0], N=1, reset_cond=labels[1]))
            test_policy.daemon = True
            test_policy.start()

            plt.ioff()
            plt.show()
        else:
            gps.test_policy(itr=labels[0], N=1, reset_cond=labels[1])

        sys.exit()

    if args.targetsetup:
        import matplotlib.pyplot as plt
        from gps.agent.ros_jaco.agent_ros_jaco import AgentROSJACO
        from gps.gui.target_setup_gui import TargetSetupGUI

        agent = hyperparams.config['agent']['type'](hyperparams.config['agent'])
        TargetSetupGUI(hyperparams.config['common'], agent)

        plt.ioff()
        plt.show()
    elif test_policy_N:
        import random
        import numpy as np
        import matplotlib.pyplot as plt

        seed = hyperparams.config.get('random_seed', 0)
        random.seed(seed)
        np.random.seed(seed)

        data_files_dir = exp_dir + 'data_files/'
        data_filenames = os.listdir(data_files_dir)
        algorithm_prefix = 'algorithm_itr_'
        algorithm_filenames = [f for f in data_filenames if f.startswith(algorithm_prefix)]
        current_algorithm = sorted(algorithm_filenames, reverse=True)[0]
        current_itr = int(current_algorithm[len(algorithm_prefix):len(algorithm_prefix) + 2])

        gps = GPSMain(hyperparams.config)
        if hyperparams.config['gui_on']:
            test_policy = threading.Thread(target=lambda: gps.test_policy(itr=current_itr, N=test_policy_N))
            test_policy.daemon = True
            test_policy.start()

            plt.ioff()
            plt.show()
        else:
            gps.test_policy(itr=current_itr, N=test_policy_N)
    else:
        import random
        import numpy as np

        seed = hyperparams.config.get('random_seed', 0)
        random.seed(seed)
        np.random.seed(seed)
        gps = GPSMain(hyperparams.config, args.quit)
        if hyperparams.config['gui_on']:
            import matplotlib.pyplot as plt

            run_gps = threading.Thread(target=lambda: gps.run(sessionid, itr_load=resume_training_itr))
            run_gps.daemon = True
            run_gps.start()
            plt.ioff()
            plt.show()
        else:
            gps.run(sessionid, itr_load=resume_training_itr)


if __name__ == "__main__":
    main()
