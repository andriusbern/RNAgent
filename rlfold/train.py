from rlfold.utils import Dataset
from rlfold.baselines import SBWrapper, get_parameters
import rlfold.environments
import os, sys, argparse, time
import rlfold.settings as settings

if __name__ == "__main__":
    w = SBWrapper('RnaDesign', 'tests').create_model()
    w.train()
