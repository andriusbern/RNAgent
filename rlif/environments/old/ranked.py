import numpy as np
import gym, time, os, random, sys
from collections import deque
from rlif.settings import ConfigManager as settings
from rlif.rna import Dataset, Solution, set_vienna_params

class RankedRnaDesign(gym.Env):
    def __init__(self, config=None, rank=None):
        
        self.config = config['environment']
        self.env_id = rank

        # Parameters
        self.randomize     = self.config['randomize']
        self.meta_learning = self.config['meta_learning']
        self.permute       = self.config['permute']
        # Stats
        self.ep   = 0
        self.r    = 0
        self.msg  = ''
        self.solved = []
        self.done = False
        self.verbose = True

        self.dataset = self.load_data()
        self.current_sequence = -1
        self.next_target_structure()
        self.prev_solution = None 
        self.boosting = False

        self.use_mlp = False
        state_size = np.shape(self.solution.get_state(reshape=self.use_mlp))

        # Ranked buffer
        self.ranked_rewards, self.running_hds, self.best = {}, {}, {}
        for target in self.dataset.sequences:
            self.ranked_rewards[target.file_id] = deque([.75], maxlen=self.config['buffer_size'])
            self.best[target.file_id] = 0.5
            self.running_hds[target.file_id] = deque([], maxlen=self.config['buffer_size'])

        self.observation_space = gym.spaces.Box(shape=state_size, low=0, high=1,dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(4)
        set_vienna_params(self.config['vienna_params'])

    def scale_reward(self, reward, hd):
        """
        Scale the reward based on the previous rewards on the same structure
        """
        try: 
            buffer = self.ranked_rewards[self.target_structure.file_id]
            best = self.best[self.target_structure.file_id]
            
            ranked_reward = np.mean(buffer) * self.config['reward_percentile']
            if reward > best:
                self.best[self.target_structure.file_id] = reward
            total = best * self.config['best_ratio'] + (1 - self.config['best_ratio']) * ranked_reward
            # total = ranked_reward

            if self.config['fixed_reward']:
                if reward > total:
                    scaled_reward = reward
                elif reward == total: # Randomize if rewards become stale
                    scaled_reward = 1 if random.random() > 0.5 else -1
                else:
                    scaled_reward = -1+reward
            else:
                scaled_reward = reward - total
            if scaled_reward < 0:
                scaled_reward *= 2

            if reward > total:
                buffer.append(reward)

            if reward == 1:     # If folded correctly
                scaled_reward += .25
            
            msg = 'R: {:.3f} | RR: {:.3f} | Best: {:.3f}  | SR: {}'.format(reward, ranked_reward, best, scaled_reward)

            return scaled_reward, msg
        except:
            return 0, ''

    def reset(self):
        """
        Reinitialize the generated sequence and/or target
        """
        self.prev_solution = self.solution
        self.ep += 1
        self.done = False

        if self.verbose:
            summary = '                                    '
            summary += 'Ep: {:6}, Seq: {:5}, Len : {:3}, Reward: {:5f}, HD: {:3}, Solved: {}  |  '.format(
                self.ep,
                self.target_structure.file_nr, 
                self.target_structure.len, 
                self.r, 
                self.solution.hd,
                len(self.solved))
            if self.solution.hd == 0 and self.solution.target.file_nr not in self.solved:
                self.solved.append(self.solution.target.file_nr)

            print(summary + self.msg, end='\r')

        if self.meta_learning:
            self.next_target_structure()

        self.solution = Solution(target=self.target_structure, config=self.config)

        return self.solution.get_state(reshape=self.use_mlp)

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
                string=None, 
                permute=self.permute,
                verbose=False,
                compute_statistics=False,
                boost=self.boosting)
            if self.config['enable_ranked']:
                self.r, self.msg = self.scale_reward(solution.r, solution.hd)
            else:
                self.r = solution.r

            self.done = True
        else:
            solution.find_next_unfilled()

        return solution.get_state(reshape=self.use_mlp), self.r, self.done, {}

    def load_data(self):
        """
        Loads a dataset
        """
        print('\n\nLOADING DATA\n\n')
        dataset = Dataset(
            dataset=self.config['dataset'],
            start=1,
            n_seqs=100,
            encoding_type=self.config.get('encoding_type'))
        
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
    env = RankedRnaDesign(get_parameters('RankedRnaDesign'))
    # env.visual_test(True)
