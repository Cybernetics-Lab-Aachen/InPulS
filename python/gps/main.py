from gps_main import GPSMain

import matplotlib as mpl
mpl.use('Qt4Agg')

import logging
import imp
import os
import os.path
import sys
import argparse
import threading
import time

# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))



def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Run the Guided Policy Search algorithm.')
    parser.add_argument('experiment', type=str,
                        help='experiment name')
    parser.add_argument('-tr', '--trainagmp', nargs='+', type=str,
                        help='train context based AGMP based on labelled samples')
    parser.add_argument('-emb', '--onlyembeddings', action='store_true', help='trains only agmp embeddings')
    parser.add_argument('-te', '--testagmp', action='store_true', help='tests context based AGMP')
    parser.add_argument('-tegcm', '--testgcm', type=int, metavar='<iteration>', help='tests context based GCM')
    parser.add_argument('-tgc', '--testgcmcond', nargs='+', type=int, help='test policy [iteration, start condition]')
    parser.add_argument('-tc', '--testcond', nargs='+', type=int, help='test policy [iteration, start condition]')
    parser.add_argument('-l', '--logging', nargs='?', const=[time.strftime("%S%M%H-%d%m%Y")], type=str,
                        help='enable labeled logging')
    parser.add_argument('-n', '--new', action='store_true',
                        help='create new experiment')
    parser.add_argument('-t', '--targetsetup', action='store_true',
                        help='run target setup')
    parser.add_argument('-r', '--resume', metavar='N', type=int,
                        help='resume training from iter N')
    parser.add_argument('-p', '--policy', metavar='N', type=int,
                        help='take N policy samples (for BADMM/MDGPS only)')
    parser.add_argument('-s', '--silent', action='store_true',
                        help='silent debug print outs')
    parser.add_argument('-q', '--quit', action='store_true',
                        help='quit GUI automatically when finished')
    args = parser.parse_args()

    exp_name = args.experiment
    resume_training_itr = args.resume
    test_policy_N = args.policy
    sessionid = args.logging

    if sessionid:
        print("Start with Logging ID: ", sessionid)
    else:
        sessionid = ''

    from gps import __file__ as gps_filepath
    gps_filepath = os.path.abspath(gps_filepath)
    gps_dir = '/'.join(str.split(gps_filepath, '/')[:-3]) + '/'
    exp_dir = gps_dir + 'experiments/' + exp_name + '/'
    hyperparams_file = exp_dir + 'hyperparams.py'

    if args.silent:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    else:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    if args.new:
        from shutil import copy

        if os.path.exists(exp_dir):
            sys.exit("Experiment '%s' already exists.\nPlease remove '%s'." %
                     (exp_name, exp_dir))
        os.makedirs(exp_dir)

        prev_exp_file = '.previous_experiment'
        prev_exp_dir = None
        try:
            with open(prev_exp_file, 'r') as f:
                prev_exp_dir = f.readline()
            copy(prev_exp_dir + 'hyperparams.py', exp_dir)
            if os.path.exists(prev_exp_dir + 'targets.npz'):
                copy(prev_exp_dir + 'targets.npz', exp_dir)
        except IOError as e:
            with open(hyperparams_file, 'w') as f:
                f.write('# To get started, copy over hyperparams from another experiment.\n' +
                        '# Visit rll.berkeley.edu/gps/hyperparams.html for documentation.')
        with open(prev_exp_file, 'w') as f:
            f.write(exp_dir)

        exit_msg = ("Experiment '%s' created.\nhyperparams file: '%s'" %
                    (exp_name, hyperparams_file))
        if prev_exp_dir and os.path.exists(prev_exp_dir):
            exit_msg += "\ncopied from     : '%shyperparams.py'" % prev_exp_dir
        sys.exit(exit_msg)

    if not os.path.exists(hyperparams_file):
        sys.exit("Experiment '%s' does not exist.\nDid you create '%s'?" %
                 (exp_name, hyperparams_file))

    hyperparams = imp.load_source('hyperparams', hyperparams_file)

    if args.trainagmp:
        # load datasets (samples, controller)
        # load tensorflow
        # train contextbased policy
        # store contextbased policy (only one per experiment!)

        import glob
        import copy
        import cPickle as pickle
        from gps.algorithm.agmp_x.agmp_training import AGMP_Training

        labels = args.trainagmp
        data_files_dir = exp_dir + 'data_files/'

        samples = []
        controller = []
        for label in labels:
            sample_list_files = glob.glob(data_files_dir + label + '_samplelist_*.pkl')
            last_itr_idx = len(sample_list_files) - 1
            if last_itr_idx == -1:
                print("no sample file available")
                return
            print(str(last_itr_idx) + " sample files available for label " + label)
            last_itr_sample_file = (data_files_dir + label + ('_samplelist_itr%02d.pkl') % last_itr_idx)
            # cleaned_sample_list_files = copy.copy(sample_list_files)
            # cleaned_sample_list_files.remove(last_itr_sample_file)
            cleaned_sample_lists = [pickle.load(open(i, 'r')) for i in sample_list_files]
            last_itr_sample = pickle.load(open(last_itr_sample_file, 'r'))
            samples.append({'data_list': cleaned_sample_lists, 'optimal_data': last_itr_sample})
            print("conditions)", len(samples[-1]['optimal_data']))
            # print("len cleaned sample list: ", len(cleaned_sample_lists))           #3          2
            # print("len last_itr_sample: ", len(last_itr_sample))                    #2          1
            # print("len cleaned sample list[-1]: ", len(cleaned_sample_lists[-1]))   #2          1

            controller_list_files = glob.glob(data_files_dir + label + '_controller_*.pkl')
            last_itr_idx = len(sample_list_files) - 1
            last_itr_controller_file = (data_files_dir + label + ('_controller_itr%02d.pkl') % last_itr_idx)
            # cleaned_controller_list_files = copy.copy(controller_list_files)
            # cleaned_controller_list_files.remove(last_itr_controller_file)
            cleaned_controller_lists = [pickle.load(open(i, 'r')) for i in controller_list_files]
            last_itr_controller = pickle.load(open(last_itr_controller_file, 'r'))
            controller.append({'data_list': cleaned_controller_lists, 'optimal_data': last_itr_controller})
            # print("len cleaned_controller_lists: ", len(cleaned_controller_lists))  #3          2
            # print("len last_itr_controller: ", len(last_itr_controller))            #2          1
            # print("len cleaned_controller_lists [-1]: ", len(cleaned_controller_lists[-1]))  #

        # raw_input("wait")
        agmp_tr = AGMP_Training(hyperparams.config, data_files_dir)
        agmp_tr.run(samples, controller)
        sys.exit()

    if args.testagmp:
        #load context based model
        #test each condition of experiment
        # 1. go to reset state
        # 2. perform motion from there
        # 3. exit
        import random
        import numpy as np
        import matplotlib.pyplot as plt
        from gps.algorithm.agmp_x.agmp_controller import AGMP_Controller

        data_files_dir = exp_dir + 'data_files/'

        seed = hyperparams.config.get('random_seed', 0)
        random.seed(seed)
        np.random.seed(seed)

        gps = GPSMain(hyperparams.config)

        agmp_policy = AGMP_Controller(hyperparams.config, data_files_dir)
        agmp_policy.reset_latent_ctr_context()
        gps._take_policy_samples(20, agmp_policy)
        sys.exit()

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
        for i in range(len(gcm_sample_lists)): gcm_sample_lists[i].get_X() # Fill in data for pickling
        gps.data_logger.pickle(
            data_files_dir + ('pol_gcm_sample_itr_%02d.pkl' % args.testgcm),
            copy.copy(gcm_sample_lists)
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
                    pn_i = idx_i/particle_number
                    pn_j = idx_j/particle_number
                    sample_z = np.array([pn_i, pn_j])
                    cur_ctr, cur_x = gcm_policy.sample_state_controller(np.reshape(sample_z,[1,-1]))
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
        import gps.algorithm.gcm.gcm_utils as gcm_utils


        labels = args.testgcmcond
        reset_cond = labels[1]

        print("reset to condition ", reset_cond)
        data_files_dir = exp_dir + 'data_files/'
        seed = hyperparams.config.get('random_seed', 0)
        random.seed(seed)
        np.random.seed(seed)
        gps = GPSMain(hyperparams.config)
        model_files_dir = data_files_dir + ('itr_%02d/' % labels[0])
        gcm_policy = GCMController(hyperparams.config, model_files_dir)
        gcm_policy.reset_ctr_context()

        # sample based on init configuration in hyperparams
        gcm_sample_lists = gps._take_policy_samples(1, gcm_policy, reset_cond)
        for i in range(len(gcm_sample_lists)): gcm_sample_lists[i].get_X()  # Fill in data for pickling
        gps.data_logger.pickle(
            data_files_dir + ('pol_gcm_sample_itr_%02d.pkl' % labels[0]),
            copy.copy(gcm_sample_lists)
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
            test_policy = threading.Thread(
                target=lambda: gps.test_policy(itr=labels[0], N=1, reset_cond = labels[1])
            )
            test_policy.daemon = True
            test_policy.start()

            plt.ioff()
            plt.show()
        else:
            gps.test_policy(itr=labels[0], N=1, reset_cond=labels[1])

        sys.exit()

    if args.targetsetup:
        try:
            import matplotlib.pyplot as plt
            from gps.agent.ros_jaco.agent_ros_jaco import AgentROSJACO
            from gps.gui.target_setup_gui import TargetSetupGUI

            agent = hyperparams.config['agent']['type'](hyperparams.config['agent'])
            TargetSetupGUI(hyperparams.config['common'], agent)

            plt.ioff()
            plt.show()
        except ImportError:
            sys.exit('ROS required for target setup.')
        except:
            sys.exit('The right agent class needs to be imported.')
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
            test_policy = threading.Thread(
                target=lambda: gps.test_policy(itr=current_itr, N=test_policy_N)
            )
            test_policy.daemon = True
            test_policy.start()

            plt.ioff()
            plt.show()
        else:
            gps.test_policy(itr=current_itr, N=test_policy_N)
    else:
        import random
        import numpy as np
        import matplotlib.pyplot as plt

        seed = hyperparams.config.get('random_seed', 0)
        random.seed(seed)
        np.random.seed(seed)
        gps = GPSMain(hyperparams.config, args.quit)
        if hyperparams.config['gui_on']:
            run_gps = threading.Thread(
                target=lambda: gps.run(sessionid, itr_load=resume_training_itr)
            )
            run_gps.daemon = True
            run_gps.start()
            plt.ioff()
            plt.show()
        else:
            gps.run(sessionid, itr_load=resume_training_itr)

if __name__ == "__main__":
    main()
