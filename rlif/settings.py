import os, sys, yaml
import numpy as np
import RNA

class ConfigManager():

    # Directories
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

    with open(os.path.join(CONFIG, 'Testing.yml'), 'r') as f:
        test_config = yaml.load(f)['environment']

    
    ############
    # GUI PARAMETERS
    ############
    scaling = [1, 1]
    param_files = [
        'rna_turner1999.par',
        'rna_turner2004.par',
        'rna_andronescu2007.par',
        'rna_langdon2018.par']

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
    ATTEMPTS = 20
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
        permutation_threshold = np.arange(1, 30))

    if sys.platform in ['linux', 'darwin']:
        fold_fn = RNA.fold
        delimiter = '/'
    elif sys.platform == 'win32':
        from rlif.rna import fold
        fold_fn = fold
        delimiter = '\\'
    else: 
        print('Unknown OS.')

    # List of best_solution trained models in ../trained_models dir
    #           num    checkpoint   boosting
    model_args =   [
        [2, '8',  False],
        [2, '8',  True],
        [3, '9',  False],
        [3, '9',  True],
        [1, '10', False],
        [1, '12', False],
        [4, '9',  False],
        [5, '11', False]]

    @staticmethod
    def get(parameter):
        return ConfigManager.__dict__[parameter] 
    
    # def set(parameter, value):
    #     ConfigManager.__dict__[parameter] = value

    def __init__(self):
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
            permutation_budget = '# of Permutations',
            permutation_radius = 'Permutation radius',
            permutation_threshold = 'Permutation threshold')
        return translate[parameter]


def get_parameters(env_name, model_path=None, config_name=None, config_location=None):
    """
    Method for getting the YAML config file of the RL model, policy and environment
    Get config by prioritizing:
        1. Specific config file: /config/[name].yml
        2. From model'solution directory (in case of loading) /trained_models/_/_/_/parameters.yml
        3. /config/[env_name].yml
        4. /config/[env_type].yml
        5. /config/defaults.yml
    """
    # if env_name is not None:
    env_type = get_env_type(env_name)
    env_params = os.path.join(ConfigManager.CONFIG, env_name+'.yml')
    if config_location is not None:
        path = config_location
    else:
        if config_name is not None:
            path = os.path.join(ConfigManager.CONFIG, config_name + '.yml')
        elif model_path is not None:
            path = os.path.join(model_path, 'config.yml')
        elif os.path.isfile(env_params):
            path = env_params
        else:
            path = os.path.join(ConfigManager.CONFIG, env_type + '.yml')

    with open(path, 'r') as f:
        config = yaml.load(f)

    # Parse some of the config for saving later
    main = config['main']
    config['policy'] = config['policies'][main['policy']]
    config['model'] = config['models'][main['model']]

    return config

def get_env_type(env_name):
    """
    Get the type of environment from the env_name string
    """

    if 'Rna' in env_name:
        return 'rna'
    else:
        try:
            gym.make(env_name)
            return 'gym'
        except:
            return None
