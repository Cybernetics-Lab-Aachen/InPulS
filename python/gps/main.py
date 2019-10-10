from gps_main import GPSMain

import logging
import imp
import os
import os.path
import sys
import argparse
import threading

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
        gps = GPSMain(hyperparams.config)
        if hyperparams.config['gui_on']:
            import matplotlib.pyplot as plt

            run_gps = threading.Thread(target=lambda: gps.run(itr_load=resume_training_itr))
            run_gps.daemon = True
            run_gps.start()
            plt.ioff()
            plt.show()
        else:
            gps.run(itr_load=resume_training_itr)


if __name__ == "__main__":
    main()
