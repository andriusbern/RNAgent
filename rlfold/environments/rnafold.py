import sys
sys.path.append('/usr/local/lib/python3.6/site-packages')
import RNA
import numpy as np
from rlfold.utils import Dataset, Solution, Sequence
import gym, time, os, random
import cv2
import rlfold.settings as settings
from rlfold.utils import colorize_nucleotides


"""
TODO:
    1. State representation 
        1. Binary matrices - row 1-2 - dots brackets, row 3-7 or 3-6 - nothing/nucleotides
        2. Simple ternary representation
"""

class RnaDesign(gym.Env):
    def __init__(self, config=None):

        self.config = config['environment']
        print(self.config)

        # Mappings
        self.mapping = {0:'A', 1:'C', 2:'G', 3:'U'}
        self.reverse_mapping = {'A':0, 'C':1, 'G':2, 'U':3}
        self.reverse_action = {0:3, 1:2, 2:1, 3:1}

        self.meta_learning = self.config['meta_learning']
        self.state = None

        if 'randomize' in self.config.keys():
            self.randomize = self.config['randomize']
        else:
            self.randomize = False
        self.current_sequence = -1

        self.use_full_state = self.config['full_state']

        # Logging 
        if 'path' not in self.config.keys():
            self.config['path'] = settings.RESULTS
        
        self.dataset = Dataset(length=self.config['seq_len'], n_seqs=self.config['seq_count'])
        self.next_target()
        self.folded_design = ''
        self.good_solutions = []
        self.prev_solution = None

        # Parameters
        self.kernel_size = self.config['kernel_size']
        self.reward_exp = self.config['reward_exp']

        # Stats
        self.ep  = 0
        self.r   = 0
        self.lhd = 500
        self.current_nucleotide = 0
        self.done = False
        self.action_space = gym.spaces.Discrete(4)

        if self.use_full_state:
            self.observation_space = gym.spaces.Box(shape=(6, self.kernel_size*2, 1), low=0, high=1)
        else:
            self.observation_space = gym.spaces.Box(shape=(self.kernel_size*2,1,1), low=0, high=1)

    def next_target(self, r=True):

        if self.randomize:
            self.target = self.dataset.sequences[random.randint(0, self.dataset.n_seqs-1)]
        else:
            self.target = self.dataset.sequences[self.current_sequence]
            self.current_sequence += 1
            if self.current_sequence >= self.dataset.n_seqs - 1:
                self.current_sequence = 0
        self.solution = Solution(self.target, self.config)

    def step(self, action):
        """
        """
        self.r = 0
        self.solution.str_action(action)
        self.solution.matrix_action(action)

        if self.solution.current_nucleotide == self.target.len - 1:
            self.solution.r = self.r = self.solution.evaluate()
            self.done = True
        else:
            self.solution.find_next_unfilled()
        self.state = self.solution.get_state()

        return self.state, self.r, self.done, {}

    def reset(self):
        """
        Initialize the sequence
        """
        self.prev_solution = self.solution 
        self.ep += 1
        self.done = False
        if self.solution.hd < self.lhd: self.lhd = self.solution.hd
        if self.solution.hd <= self.config['write_threshold']:
            self.solution.write_solution()
            self.good_solutions.append(self.solution)
        print('Ep: {:6}, Seq: {:5}, Len : {:3}, Reward: {:5f}, HD: {:3} ({:3})'.format(self.ep, self.target.file_nr, self.target.len, self.r, self.solution.hd, self.lhd), end='\r')
        
        self.state = self.solution.get_state()
        if self.meta_learning:
            self.next_target()
        else:
            self.solution = Solution(self.target, self.config)
    
        return self.state

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
                print(self.target.seq)
                print(self.solution.string, end='\r\r\r') #' ', self.target.seq, '\n ',
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
                print(self.target.seq)
                print('Bad: {}'.format(self.current_index))
                # del self.dataset.sequences[self.current_index - 1]
            self.next_target()

    def random_sampling_test(self,runs=1000):
        for _ in range(runs):
            self.reset()
            while not self.done:
                action = self.action_space.sample()
                _, _, self.done, _ = self.step(action)

    def visual_test(self):
        name = 'Visual Test'
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 1000, 200)
        
        for _ in range(20):
            self.reset()
            while not self.done:
                action = self.action_space.sample()
                image, _, self.done, _ = self.step(action)
                if self.use_full_state:
                    image *= 120
                    # image[:, self.solution.current_nucleotide] = 100
                    # image[:,start] = 255
                    # image[:,start+self.solution.kernel_size*2] = 255
                    print(colorize_nucleotides(self.solution.string), end='\r')
                im = np.asarray(image, dtype=np.uint8)
                cv2.imshow(name, im)
                cv2.waitKey(100)
            print(''.join([' '] * 500))
            self.solution.summary(True)
            print('\n')
        cv2.destroyWindow(name)

    def speed_test(self):
        """
        """

if __name__ == "__main__":
    env = RnaDesign()
