import sys
print(sys.path)
try:
    sys.path.remove('/usr/local/lib/python3.6/site-packages')
except:
    pass
from rlfold.definitions import Dataset
from rlfold.baselines import SBWrapper, get_parameters
import rlfold.environments
import os, argparse, time
import rlfold.settings as settings

if __name__ == "__main__":
    w = SBWrapper('RankedRnaDesign', 'baby').load_model(16)
    w._tensorboard()
    w.train()