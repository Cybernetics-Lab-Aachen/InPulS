from gps_main import GPSMain

import logging
import imp
import os
import os.path
import sys
import argparse
import random
import numpy as np

# Add gps/python to path so that imports work. Replace backslashes for windows compability
sys.path.append('/'.join(str.split(__file__.replace('\\', '/'), '/')[:-2]))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='Run the Guided Policy Search algorithm.')
    parser.add_argument('experiment', type=str, help='experiment name')
    args = parser.parse_args()

    exp_name = args.experiment

    from gps import __file__ as gps_filepath
    gps_filepath = os.path.abspath(gps_filepath).replace('\\', '/')  # Replace backslashes for windows compability
    gps_dir = '/'.join(str.split(gps_filepath, '/')[:-3]) + '/'
    exp_dir = gps_dir + 'experiments/' + exp_name + '/'
    hyperparams_file = exp_dir + 'hyperparams.py'

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARN)

    if not os.path.exists(hyperparams_file):
        sys.exit("Experiment '%s' does not exist.\nDid you create '%s'?" % (exp_name, hyperparams_file))

    hyperparams = imp.load_source('hyperparams', hyperparams_file)

    seed = hyperparams.config.get('random_seed', 0)
    random.seed(seed)
    np.random.seed(seed)
    gps = GPSMain(hyperparams.config)
    gps.run()
