from rlif.rna import Dataset
from rlif.learning import Trainer, get_parameters
import rlif.environments
import os, sys, argparse, time
from rlif.settings import ConfigManager as settings
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num', type=int)
    parser.add_argument('-s', '--subdir', type=str)
    parser.add_argument('-c', '--checkpoint', type=str, default='')
    parser.add_argument('-t', '--timeout', type=int, default=60)
    args = parser.parse_args()

    trained_model = Trainer('RnaDesign', subdir=args.subdir).load_model(num=args.num, checkpoint=args.checkpoint)
    trained_model._tensorboard()
    trained_model.test_runner.evaluate(time_limit=args.timeout, permute=True, verbose=True)
