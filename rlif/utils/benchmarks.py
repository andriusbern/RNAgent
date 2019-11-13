from rlif.rna import Dataset
from rlif.learning import Trainer, get_parameters
import rlif.environments
import os, sys, argparse, time
from rlif.settings import ConfigManager as settings
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    msg = """
    Runs a test on benchmark datasets:
    
    
    """
    print(msg)
    parser.add_argument('-t', '--timeout', type=int, default=60)
    args = parser.parse_args()

    trained_model = Trainer('RnaDesign').load_model(1, checkpoint='10', t_env=True)
    trained_model.test_runner.evaluate(time_limit=args.timeout, permute=True, verbose=True)
