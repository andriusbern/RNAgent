import numpy as np
import gym, time, os, random, sys
from rlif.settings import ConfigManager as settings
from rlif.rna import Dataset, Solution
 
class RnaMultiDesign(gym.Env):
    def __init__(self, config=None, rank=None):
        
        self.config = config['environment']
        self.env_id = rank

        # Parameters
        self.randomize     = self.config['randomize']
        self.meta_learning = self.config['meta_learning']
        self.permute       = self.config['permute']
        # Stats
        self.ep   = 0
        self.done = False
        self.verbose = True

        self.dataset = self.load_data()
        self.current_sequence = 0
        self.next_target_structure()
        self.prev_solution = None 
        self.boosting = False
        self.testing_mode = True
        self.use_mlp = True

        states = [7 for x in range(self.config['kernel_size']*2)] + [5 for x in range(self.config['kernel_size']*2)]
        self.observation_space = gym.spaces.MultiDiscrete(states)
        self.action_space = gym.spaces.Discrete(4)

    def reset(self):
        """
        Reinitialize the generated sequence and/or target
        """
        self.prev_solution = self.solution
        self.ep += 1
        self.done = False

        if self.verbose:
            summary = '                                    '
            summary += 'Ep: {:6}, Seq: {:5}, Len : {:3}, Reward: {:5f}, HD: {:3}'.format(
                self.ep,
                self.target_structure.file_nr, 
                self.target_structure.len, 
                self.solution.r, 
                self.solution.hd)
            print(summary, end='\r')

        if self.meta_learning:
            self.next_target_structure()

        self.solution = Solution(target=self.target_structure, config=self.config)

        return self.solution.get_state(reshape=self.use_mlp)

    def step(self, action):
        """
        Generate a nucleotide at the current location
        """
        solution = self.solution
        solution.str_action(action)
        solution.md_action(action)

        if solution.index == self.target_structure.len - 1:
            solution.evaluate(
                string=None, 
                permute=self.permute,
                verbose=False,
                compute_statistics=self.testing_mode,
                boost=self.boosting)

            self.done = True
        else:
            solution.find_next_unfilled()

        return solution.get_state(reshape=self.use_mlp), solution.r, self.done, {}

    def load_data(self):
        """
        Loads a dataset
        """
        encoding_type = self.config.get('encoding_type')
        dataset = Dataset(
            length=self.config['seq_len'],
            n_seqs=self.config['seq_count'],
            encoding_type=encoding_type)
        
        return dataset

    def set_data(self, data):
        self.dataset = data
        self.next_target_structure()

    def next_target_structure(self):
        """
        Get the next target secondary structure from the dataset
        """
        if self.randomize:
            index = random.randint(0, self.dataset.n_seqs-1)
        else:
            index = self.current_sequence = (self.current_sequence + 1) % len(self.dataset.sequences)
        self.target_structure = self.dataset.sequences[index]
        self.solution = Solution(target=self.target_structure, config=self.config)

if __name__ == "__main__":
    from rlif.learning import Trainer, get_parameters
    env = RnaDesign(get_parameters('RnaDesign'))
    # env.visual_test(True)
