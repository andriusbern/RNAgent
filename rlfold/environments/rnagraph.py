import sys
# sys.path.append('/usr/local/lib/python3.6/site-packages')
import numpy as np
from rlfold.utils import Dataset, GraphSolution, Sequence
import gym, time, os, random
import cv2
import rlfold.settings as settings
from rlfold.utils import colorize_nucleotides
from gensim.models.doc2vec import Doc2Vec
from rlfold.utils import WeisfeilerLehmanMachine

import networkx as nx
import matplotlib.pyplot as plt
plt.ion()

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

        
        self.next_target()
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
        
    def next_target(self):
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
            self.next_target()
        else:
            self.solution = GraphSolution(self.target_structure, self.config)
        self.state = self.solution.get_features()
        machine = WeisfeilerLehmanMachine(self.solution.current_subgraph, self.state, 2)
        processed = machine.extracted_features
        embedding = self.embedder.infer_vector(processed)
    
        return np.array(embedding)

    # Misc
    def test(self, s=False, b=False):
        """
        Test the current seq for errors
        """
        self.reset()
        while not self.done:
            action = self.action_space.sample()
            state, _, self.done, _ = self.step(action)
            if s:
                print(state, end='\r')
                time.sleep(0.01)

    def test_run(self):
        for _ in range(1):
            self.reset()
            while not self.done:
                action = self.action_space.sample()
                _, _, self.done, _ = self.step(action)
                print(self.target_structure.seq)
                print(self.solution.string, end='\r\r\r') #' ', self.target_structure.seq, '\n ',
                # time.sleep(0.05)
                input()
            self.next_target()
        self.reset()
    
    def parse_data(self):
        """
        Tests whether all the sequences are correctly parsed
        """
        for i in range(self.dataset.n_seqs - 1):
            print(i)
            try:
                self.test()
            except KeyError:
                print(self.target_structure.seq)
                print('Bad: {}'.format(self.solution.index))
                # del self.dataset.sequences[self.solution.index - 1]
            self.next_target()

    def random_sampling_test(self,runs=1000):
        for _ in range(runs):
            self.reset()
            while not self.done:
                action = self.action_space.sample()
                _, _, self.done, _ = self.step(action)

    def visual_test(self, pause=True):
        name = 'Visual Test'
        # cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(name, 1000, 200)
        self.dataset = Dataset(dataset='eterna', start=1, n_seqs=10)
        mappy = {'A':'green', 'U':'blue', 'G':'red', 'C':'purple', 'O':'black', '-':'gray'}
        step = 0
        for _ in range(20):
            self.reset()
            while not self.done:
                step += 1
                print(step)
                action = self.action_space.sample()
                state, _, self.done, _ = self.step(action)
                print(state)
                print(colorize_nucleotides(self.solution.string), end='\r')
                subgraph = self.solution.current_subgraph
                features = self.solution.get_features()
                machine = WeisfeilerLehmanMachine(subgraph, features, 2)
                processed = machine.extracted_features
                embedding = self.embedder.infer_vector(processed)

                plt.figure(1)
                plt.cla()
                plt.ylim([-.8, .8])
                plt.plot(embedding)
                plt.show()

                plt.figure(2)
                feats = [mappy[feat] for feat in features.values()]
                pos = nx.spring_layout(subgraph)
                # subgraph = nx.Graph()
                # nx.set
                nx.draw(subgraph, pos, node_color=feats)
                plt.show(); plt.pause(0.01)
                if pause: input()
                plt.cla()
            print(''.join([' '] * 500))
            self.solution.summary(True)
            print('\n')

    def speed_test(self):

        """
        """
        for _ in range(1000):
            self.reset()
            while not self.done:
                action = self.action_space.sample()
                image, _, self.done, _ = self.step(action)

                    # image[:, self.solution.current_nucleotide] = 100
                    # image[:,start] = 255
                    # image[:,start+self.solution.kernel_size*2] = 255

if __name__ == "__main__":
    from rlfold.baselines import SBWrapper, get_parameters
    config = get_parameters('RnaGraphDesign')
    config['environment']['seq_count'] = 1
    config['environment']['seq_len'] = [105, 106]
    env = RnaGraphDesign(config)
    env.visual_test()

