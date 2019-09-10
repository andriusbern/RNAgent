import sys
# sys.path.append('/usr/local/lib/python3.6/site-packages')
import numpy as np
from rlfold.definitions import Dataset, Solution, Sequence#, Tester
import gym, time, os, random
import cv2
import rlfold.settings as settings
from rlfold.definitions import colorize_nucleotides
from rlfold.interface import create_browser, show_rna

"""
TODO:
    1. State representation
        1. Binary matrices - row 1-2 - dots brackets, row 3-7 or 3-6 - nothing/nucleotides
        2. Simple ternary representation
"""
 
class RnaDesign(gym.Env):
    def __init__(self, config=None, rank=None):
        
        self.config = config['environment']
        self.env_id = rank
        self.meta_learning = self.config['meta_learning']


        # Logging
        if 'path' not in self.config.keys():
            self.config['path'] = settings.RESULTS

        self.randomize = self.config['randomize']
        encoding_type = self.config.get('encoding_type')
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
                        encoding_type=encoding_type)

                    sequences += data.sequences
                self.dataset = Dataset(sequences=sequences)

            else:
                self.dataset = Dataset(
                    length=self.config['seq_len'],
                    n_seqs=self.config['seq_count'],
                    encoding_type=encoding_type)

        self.current_sequence = 0
        self.next_target_structure()
        self.prev_solution = None
        self.permute = self.config['permute']

        # Parameters
        self.use_full_state = self.config['full_state']
        self.kernel_size = self.config['kernel_size']
        self.reward_exp = self.config['reward_exp']

        # Stats
        self.ep  = 0
        self.r   = 0
        self.lhd = 500
        self.done = False
        self.verbose = True
        state_size = np.shape(self.solution.get_state())

        if self.use_full_state:
            self.observation_space = gym.spaces.Box(shape=(state_size[0], self.kernel_size*2, 1), low=0, high=1,dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Box(shape=(self.kernel_size*2,1,1), low=0, high=1, dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(4)

    def next_target_structure(self):
        """
        Get the next target secondary structure from the dataset
        """
        if self.randomize:
            self.target_structure = self.dataset.sequences[random.randint(0, self.dataset.n_seqs-1)]
        else:
            self.current_sequence = (self.current_sequence + 1) % len(self.dataset.sequences)
            self.target_structure = self.dataset.sequences[self.current_sequence]

        self.solution = Solution(self.target_structure, self.config)

    def step(self, action):
        """
        Generate a nucleotide at the current location
        """
        solution = self.solution
        self.r = 0
        solution.str_action(action)
        solution.matrix_action(action)

        if solution.index == self.target_structure.len - 1:
            solution.r, _, _ = self.solution.evaluate(
                string=solution.string, 
                permute=self.permute,
                verbose=False)

            self.r = solution.r
            self.done = True
        else:
            shaped = self.shape_reward()
            # self.r += self.shape_reward()
            # print(shaped)
            solution.find_next_unfilled()

        if self.r == 1:
            self.r += .5

        return solution.get_state(), self.r, self.done, {}

    def reset(self):
        """
        Reinitialize the generated sequence and/or target
        """
        self.prev_solution = self.solution
        self.ep += 1
        self.done = False

        if self.solution.hd < self.lhd: self.lhd = self.solution.hd
        # if self.solution.hd <= self.config['write_threshold']:
        #     self.solution.write_solution()

        if self.verbose:
            summary = '                                    '
            summary += 'Ep: {:6}, Seq: {:5}, Len : {:3}, Reward: {:5f}, HD: {:3} ({:3})'.format(
                        self.ep,
                        self.target_structure.file_nr, 
                        self.target_structure.len, 
                        self.r, 
                        self.solution.hd, 
                        self.lhd)
                    
            print(summary, end='\r')

        if self.meta_learning:
            self.next_target_structure()
        else:
            self.solution = Solution(target=self.target_structure, config=self.config)

        return self.solution.get_state()

    def shape_reward(self):
        """
        """
        reward = 0
        index  = self.solution.index
        string = self.solution.string
        # Internal loops

        if self.target_structure.markers[index] == 'I':
            if self.target_structure.markers[index-1] != 'I' and string[index] == 'G':
                reward += 0.001
            if self.target_structure.markers[index+1] != 'I' and string[index] == 'A':
                reward += 0.001

        if self.target_structure.markers[index] == 'S':
            if self.target_structure.markers[index-1] != 'S' or self.target_structure.markers[index+1] != 'S':
                if string[index] == 'G' or string[index] == 'C':
                    reward += 0.001
            else:
                if string[index] == 'A' or string[index] == 'U':
                    reward += 0.001

        # if self.target_structure.markers[index] == 'M':
        #     if self.target_structure[index-1] != 'I' and string[index] == 'A':
        #         reward += 0.01
        #     if self.target_structure[index+1] != 'I' and string[index] == 'G':
        #         reward += 0.01
        # Add later

        if self.target_structure.markers[index] == 'H':
            if self.target_structure.markers[index-1] != 'H' and string[index] == 'G':
                reward += 0.001
            if self.target_structure.markers[index+1] != 'H' and string[index] == 'A':
                reward += 0.001
        return reward

    ################
    ### Test methods

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
            self.next_target_structure()
        self.reset()

    def random_sampling_test(self,runs=1000):
        for _ in range(runs):
            self.reset()
            while not self.done:
                action = self.action_space.sample()
                _, _, self.done, _ = self.step(action)

    def visual_test(self, pause=False):
        name = 'Visual Test'
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 1000, 200)

        # driver = create_browser('double')

        for _ in range(20):
            self.reset()
            # show_rna(self.target_structure.seq, None, driver, 0, 'double')
            print(self.target_structure.markers)
            print(self.target_structure.seq)
            while not self.done:
                action = self.action_space.sample()
                image, _, self.done, _ = self.step(action)

                if self.use_full_state:
                    image *= 120
                    print(colorize_nucleotides(self.solution.string), end='\r')

                im = np.asarray(image, dtype=np.uint8)
                cv2.imshow(name, im); cv2.waitKey(1)
                if pause: input()

            print(''.join([' '] * 500))
            self.solution.summary(True)
            print('\n')
            # show_rna(self.solution.folded_design.seq, self.solution.string, driver, 1, 'double')
            if pause: input()
        cv2.destroyWindow(name)

    def speed_test(self):

        """
        """
        for _ in range(1000):
            self.reset()
            while not self.done:
                action = self.action_space.sample()
                image, _, self.done, _ = self.step(action)


if __name__ == "__main__":
    from rlfold.baselines import SBWrapper, get_parameters
    env = RnaDesign(get_parameters('RnaDesign'))
    # env.speed_test()
    env.visual_test(True)
