from rlfold.definitions import Dataset
from rlfold.baselines import SBWrapper, get_parameters
import rlfold.environments
import os, sys, argparse, time
import rlfold.settings as settings
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num', type=int)
    parser.add_argument('-s', '--subdir', type=str)
    parser.add_argument('-t', '--timeout', type=int, default=60)
    # parser.add_argument('-r', '--radius', type=int, default=1)
    # parser.add_argument('-b', '--budget', type=int, default=20)
    # parser.add_argument('-v', '--threshold', type=int, default=5)
    args = parser.parse_args()


    trained_model = SBWrapper('RnaDesign', subdir=args.subdir).load_model(num=args.num)

    # Modify parameters
    # trained_model.config['environment']['permutation_threshold'] = args.threshold
    # trained_model.config['environment']['permutation_radius'] = args.radius
    # trained_model.config['environment']['permutation_budget'] = args.budget

    # trained_model._tensorboard()
    trained_model.test_runner.evaluate(time_limit=args.timeout, permute=True, verbose=True)
