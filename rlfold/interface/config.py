import sys, os, random

import numpy as np


class ConfigManager():

    # Directories
    SRC_DIR = os.path.dirname(os.path.realpath(__file__))
    MAIN_DIR = os.path.abspath(os.path.join(SRC_DIR, os.pardir))
    DATA = os.path.join(SRC_DIR, 'data')
    ICONS = os.path.join(MAIN_DIR, 'icons')
    AUDIO = os.path.join(DATA, 'audio')
    IMAGES = os.path.join(DATA, 'images')
    MODELS = os.path.join(SRC_DIR, 'models')

    ############
    # Parameters
    ############

    VIENNA_PARAMS = 0
    temperature = 37
    dangles = 2
    noGU = 0
    no_closingGU = 0
    uniq_ML = 0

    permutation_budget = 5
    permutation_radius = 2
    permutation_threshold = 5
    TIME = 60
    WORKERS = 1
    ATTEMPTS = 10
    N_SOLUTIONS = 50
    VERBOSITY = 0
    MODEL_NR = 0

    # Value ranges
    ranges = dict(
        temperature = np.arange(0, 100, 0.1),
        dangles = [0, 1, 2, 3],
        ATTEMPTS = np.arange(-1, 1000, 1),
        N_SOLUTIONS = np.arange(1, 500, 1),
        TIME = np.arange(-1, 3600, 10),
        WORKERS = [x for x in range(1, os.cpu_count()*2)],
        permutation_budget = np.arange(0, 50),
        permutation_radius = np.arange(1,5),
        permutation_threshold = np.arange(1, 30),
        )

    @staticmethod
    def get(parameter):
        return ConfigManager.__dict__[parameter] 
    
    def set(parameter, value):
        ConfigManager.__dict__[parameter] = value
        # setattr(locals(), parameter, value)
        # setattr(globals(), parameter, value)

    def __init__(self):
        pass
    
    def get_defaults(self):
        pass

    def write_config(self, path):
        pass

    def read_config(self, path):
        pass

    @staticmethod
    def translate(parameter):
        translate = dict(
            VIENNA_PARAMS = 'Fold parameters',
            temperature = 'Temperature',
            dangles = 'Dangling ends',
            noGU = 'No GU pairs',
            no_closingGU = 'Closing GU pairs',
            uniq_ML = 'Multiloop decomposition',
            ATTEMPTS = '# Attempts',
            N_SOLUTIONS = '# Solutions',
            TIME = 'Time limit',
            WORKERS = 'Parallel workers',
            permutation_budget = '# Permutations',
            permutation_radius = 'Permutation radius',
            permutation_threshold = 'Permutation threshold',
        )
        return translate[parameter]