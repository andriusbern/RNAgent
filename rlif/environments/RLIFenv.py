import numpy as np
import gym, time, os, random, sys
from rlif.settings import ConfigManager as settings
from rlif.rna import Dataset, Solution, DotBracket
 
class RnaDesign(gym.Env):
    def __init__(self, config=None, rank=None):
        self.config = config
        self.env_id = rank

        # Parameters
        self.randomize     = self.config['randomize']
        self.meta_learning = self.config['meta_learning']
        self.permute       = self.config['permute']
        self.verbose       = self.config['verbose']
        self.boosting = False
        self.use_mlp  = False
        self.testing_mode = True

        # Stats
        self.ep   = 0
        self.done = False

        # Data
        self.current_sequence = -1
        self.dataset = self.load_data()
        self.next_target_structure()
        self.prev_solution = None

        state_size = np.shape(self.solution.get_state(reshape=self.use_mlp))
        self.observation_space = gym.spaces.Box(shape=state_size, low=0, high=1,dtype=np.uint8)
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
        solution.insert_nucleotide(action)

        if solution.index < self.target_structure.len - 1:
            solution.find_next_unfilled()
        else:
            self.done = True
            solution.evaluate(
                reward=not self.testing_mode,
                permute=self.permute,
                verbose=False,
                compute_statistics=self.testing_mode,
                boost=self.boosting)
        state, reward = solution.get_state(reshape=self.use_mlp), solution.r

        return state, reward, self.done, {}

    def next_target_structure(self):
        """
        Get the next target secondary structure from the dataset
        """
        if self.randomize:
            index = random.randint(0, self.dataset.n_seqs-1)
        else:
            l = len(self.dataset.sequences) if len(self.dataset.sequences) > 0 else 1
            index = self.current_sequence = (self.current_sequence + 1) % l
        self.target_structure = self.dataset.sequences[index]
        self.solution = Solution(target=self.target_structure, config=self.config)

    def load_data(self):
        """
        Loads a dataset
        """
        dataset = Dataset(
            length=self.config['seq_len'],
            n_seqs=self.config['seq_count'],
            encoding_type=self.config['encoding_type'])
        
        return dataset

    def set_data(self, data):
        self.current_sequence = -1
        self.dataset = data
        self.next_target_structure()

    def set_sequence(self, sequence):
        self.dataset.sequences = [DotBracket(sequence, encoding_type=2)]
        self.next_target_structure()

if __name__ == "__main__":
    from rlif.learning import Trainer, get_parameters
    env = RnaDesign(get_parameters('RnaDesign'))
