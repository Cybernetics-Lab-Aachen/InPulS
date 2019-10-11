from gps_main import GPSMain

import logging
import imp
import os
from pathlib import Path
import sys
import argparse
import random
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
