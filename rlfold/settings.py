import os
import sys

MAIN_DIR = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(MAIN_DIR, 'data')
TRAINED_MODELS = os.path.join(MAIN_DIR, 'trained_models')
CONFIG = os.path.join(MAIN_DIR, 'config')
RESULTS = os.path.join(MAIN_DIR, 'results')
VIENNA_DIR = r'C:\Program Files (x86)\ViennaRNA Package'
ICONS = os.path.join(MAIN_DIR, 'interface', 'icons')

CURRENT_SOLUTION = 0

os = sys.platform
if os == 'win32':
    fold_engine = 'cli'
    delimiter = '\\'

if os == 'linux':
    fold_engine = 'python'
    delimiter = '/'


model_dict = {
    '0': ['experiment4', 2, ''], # Best fo sho, low U
    '1': ['t238', 1, '10'], # 
    '2': ['e238', 2, '11'],
    '3': ['experiment5', 1, '12'],
    '4': ['experiment4', 2, '8'],
    '5': ['long', 1, '56'],
    '6': ['trit', 1, '20'], # bad
    # '2': ['l', 1, '20'], # very bad
    '7': ['l2', 1, '32'],
    '8': ['latest', 1, '39'],
    '9': ['latest', 2, '31'],
    '10': ['experiment6', 2, '19'],
    }

def get_solution_id():
    """
    Function to increment the current solution id
    (because of multiple workers)
    """
    global CURRENT_SOLUTION
    CURRENT_SOLUTION += 1
    return CURRENT_SOLUTION


class ParameterContainer(dict):
    """
    Modified dict - can access keys with dict.key notation instead of dict['key']
    """

    def __getattribute__(self, item):
        return self[item]
