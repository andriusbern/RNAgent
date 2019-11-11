import os, sys
import numpy as np

try:
    sys.path.remove('/usr/local/lib/python3.6/site-packages')
except:
    pass

import RNA


class ConfigManager():
    # Directories

    OPERATING_SYSTEM = sys.platform

    if OPERATING_SYSTEM == 'linux':
        fold_fn = RNA.fold
        delimiter = '/'
        
    elif OPERATING_SYSTEM == 'win32':
        from rlif.rna import fold
        fold_fn = fold
        delimiter = '\\'

    MAIN_DIR = os.path.dirname(os.path.realpath(__file__))
    PARENT_DIR = os.path.abspath(os.path.join(MAIN_DIR, os.pardir))
    DATA = os.path.join(MAIN_DIR, 'data')
    ICONS = os.path.join(MAIN_DIR, 'interface', 'icons')
    TRAINED_MODELS = os.path.join(MAIN_DIR, 'trained_models')
    RESULTS = os.path.join(MAIN_DIR, 'results')
    UTILS = os.path.join(MAIN_DIR, 'utils')
    DISPLAY = os.path.join(UTILS, 'html')
    PARAMETERS = os.path.join(UTILS, 'parameters')
    CONFIG = os.path.join(MAIN_DIR, 'config')


    ############
    # Parameters
    ############
    CURRENT_SOLUTION = 0
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
    ATTEMPTS = 100
    N_SOLUTIONS = 50
    VERBOSITY = 0
    MODEL_NR = 0

    model_dict = {
        '0': ['experiment4', 2, ''], # Best fo sho, low U
        '12': ['experiment4', 1, '10'], # Most solved
        '13': ['experiment5', 1, '5'],
        '1': ['t238', 1, '8'], # 
        '2': ['experiment4', 2, '5'], # Could have more U content, works well on benches
        '3': ['e238', 2, '11'], 
        '4': ['experiment5', 1, '12'],
        '5': ['experiment4', 2, '8'],
        '6': ['long', 1, '56'],
        '7': ['trit', 1, '27'], # bad
        '8': ['l2', 1, '32'],
        '9': ['latest', 1, '39'],
        '10': ['latest', 2, '31'],
        '11': ['experiment6', 2, '19'],
    }
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
            VIENNA_PARAMS = 'Energy Parameters',
            temperature = 'Temperature',
            dangles = 'Dangling ends',
            noGU = 'No GU pairs',
            no_closingGU = 'No closing GU pairs',
            uniq_ML = 'Multiloop decomp',
            ATTEMPTS = '# Attempts',
            N_SOLUTIONS = '# Solutions',
            TIME = 'Time limit',
            WORKERS = 'Parallel workers',
            permutation_budget = '# Permutations',
            permutation_radius = 'Permutation radius',
            permutation_threshold = 'Permutation threshold')
        return translate[parameter]
    
    def get_solution_id():
        """
        Function to increment the current solution id
        (because of multiple workers)
        """
        CURRENT_SOLUTION = 1
        return CURRENT_SOLUTION

class ParameterContainer(dict):
    """
    Modified dict - can access keys with dict.key notation instead of dict['key']
    """

    def __getattribute__(self, item):
        return self[item]
