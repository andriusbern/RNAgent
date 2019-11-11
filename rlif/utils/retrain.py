import sys
from rlif.learning import Trainer, get_parameters
import rlif.environments
from rlif.settings import ConfigManager as settings

if __name__ == "__main__":
    w = Trainer('RankedRnaDesign', 'tests').load_model(16)
    w._tensorboard()
    w.train()