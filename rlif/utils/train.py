import sys
print(sys.path)
try:
    sys.path.remove('/usr/local/lib/python3.6/site-packages')
except:
    pass
from rlif.rna import Dataset
from rlif.learning import Trainer, get_parameters
import rlif.environments
import os, argparse, time
from rlif.settings import ConfigManager as settings

if __name__ == "__main__":
    w = Trainer('RnaDesign', 'final').create_model()
    w._tensorboard()
    w.train()