import os, sys

MAIN_DIR = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(MAIN_DIR, 'data')
TRAINED_MODELS = os.path.join(MAIN_DIR, 'trained_models')
CONFIG = os.path.join(MAIN_DIR, 'config')
RESULTS = os.path.join(MAIN_DIR, 'results')
VIENNA_DIR = r'C:\Program Files (x86)\ViennaRNA Package'

CURRENT_SOLUTION = 0

os = sys.platform
if os == 'win32':
   fold_engine = 'cli'
   delimiter = '\\'

if os == 'linux':
   fold_engine = 'python'
   delimiter = '/'

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