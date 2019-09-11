import sys
# sys.path.append('/usr/local/lib/python3.6/site-packages')
import numpy as np
from rlfold.definitions import Dataset, GraphSolution, Sequence
import gym, time, os, random
import rlfold.settings as settings
from rlfold.definitions import colorize_nucleotides
from gensim.models.doc2vec import Doc2Vec
from rlfold.utils import WeisfeilerLehmanMachine

import networkx as nx
import matplotlib.pyplot as plt

"""
TODO:
    1. State representation 
        1. Binary matrices - row 1-2 - dots brackets, row 3-7 or 3-6 - nothing/nucleotides
        2. Simple ternary representation
"""

class RnaGraphDesign(gym.Env):
    def __init__(self, config=None, rank=None):

        self.config = config['environment']
        self.meta_learning = self.config['meta_learning']
        self.env_id = rank
        self.state = None
        self.validator = None

        if 'randomize' in self.config.keys():
            self.randomize = self.config['randomize']
        else:
            self.randomize = False
        self.current_sequence = -1

        # Logging 
        if 'path' not in self.config.keys():
            self.config['path'] = settings.RESULTS
        datasets = ['rfam_learn_test', 'rfam_learn_validation', 'rfam_taneda', 'eterna']
        if self.config.get('test_set') is not None:
            sequences = []

            # Migrate to a function
            if self.config['test_set']:
                for dataset in datasets:
                    n_seqs = 29 if dataset=='rfam_taneda' else 100
                    data = Dataset(
                        dataset=dataset, 
                        start=1, 
                        n_seqs=n_seqs, 
                        )

                    sequences += data.sequences
                self.dataset = Dataset(sequences=sequences)

            else:
                self.dataset = Dataset(
                    length=self.config['seq_len'],
                    n_seqs=self.config['seq_count'])

        
        self.next_target_structure()
        self.folded_design = ''
        self.good_solutions = []
        self.prev_solution = None

        # Stats
        self.ep  = 0
        self.r   = 0
        self.lhd = 500
        self.current_nucleotide = 0
        self.done = False
        self.embedder = Doc2Vec.load(os.path.join(settings.MAIN_DIR, 'utils', 'train500_4each512'))
        self.permute = self.config['permute']

        self.observation_space = gym.spaces.Box(shape=(512,), low=-2., high=2.,dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)
        
    def next_target_structure(self):
        """
        Get the next target_structure secondary structure in the dataset
        """
        if self.randomize:
            self.target_structure = self.dataset.sequences[random.randint(0, self.dataset.n_seqs-1)]
        else:
            self.current_sequence += 1
            if self.current_sequence >= self.dataset.n_seqs - 1:
                self.current_sequence = 0
            try:
                self.target_structure = self.dataset.sequences[self.current_sequence]
            except:
                self.current_sequence = 0
                self.target_structure = self.dataset.sequences[self.current_sequence]
            
        self.solution = GraphSolution(self.target_structure, self.config)

    def step(self, action):
        """
        """
        reward = 0
        self.solution.str_action(action)
        self.solution.graph_action(action)

        if self.solution.index == self.target_structure.len - 1:
            reward, _ = self.solution.evaluate(self.solution.string, permute=self.permute)
            self.done = True
        else:
            self.solution.find_next_unfilled()
        self.r = reward
        self.solution.get_new_subgraph()
        self.state = self.solution.get_features()
        machine = WeisfeilerLehmanMachine(self.solution.current_subgraph, self.state, 2)
        processed = machine.extracted_features
        embedding = self.embedder.infer_vector(processed)
        return np.array(embedding), reward, self.done, {}

    def reset(self):
        """
        Initialize the sequence
        """
        self.prev_solution = self.solution 
        self.ep += 1
        self.done = False
        if self.solution.hd < self.lhd: self.lhd = self.solution.hd
        if self.solution.hd <= self.config['write_threshold']:
            # self.solution.write_solution()
            self.good_solutions.append(self.solution)
        print('                                    Ep: {:6}, Seq: {:5}, Len : {:3}, Reward: {:5f}, HD: {:3} ({:3})'
            .format(self.ep, self.target_structure.file_nr, self.target_structure.len, self.r, self.solution.hd, self.lhd), end='\r')
        
        if self.meta_learning:
            self.next_target_structure()
        else:
            self.solution = GraphSolution(self.target_structure, self.config)
        self.state = self.solution.get_features()
        machine = WeisfeilerLehmanMachine(self.solution.current_subgraph, self.state, 2)
        processed = machine.extracted_features
        embedding = self.embedder.infer_vector(processed)
    
        return np.array(embedding)


if __name__ == "__main__":
    from rlfold.baselines import SBWrapper, get_parameters
    config = get_parameters('RnaGraphDesign')
    config['environment']['seq_count'] = 1
    config['environment']['seq_len'] = [105, 106]
    env = RnaGraphDesign(config)
    env.visual_test()

