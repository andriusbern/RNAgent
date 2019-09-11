import numpy as np
import copy, random, os, datetime
from rlfold.definitions import Sequence, hamming_distance
from rlfold.definitions import colorize_nucleotides, highlight_mismatches
from rlfold.interface import create_browser, show_rna
import rlfold.settings as settings

if settings.os == 'linux':
    import RNA
    RNA.cvar.dangles = 3
    RNA.cvar.noGU = 0
    fold_fn = RNA.fold
    
elif settings.os == 'win32':
    from .vienna import fold
    fold_fn = fold

class Solution(object):
    """
    Class for logging generated nucleotide sequence solutions
    """
    def __init__(self, target, config):

        self.config = config
        self.target = target

        self.str = []
        self.use_full_state = self.config['full_state']
        if 'use_padding' in self.config.keys():
            self.use_padding = self.config['use_padding']
        else:
            self.use_padding = True

        # Statistics
        self.solution_id = settings.get_solution_id()
        self.hd = 100
        self.fe = 0
        self.reward = 0
        self.folded_design = ''
        self.reward_exp = config['reward_exp']
        self.kernel_size = config['kernel_size']
        self.index = 0
        self.padded_index = self.kernel_size

        self.mapping = {0:'AU', 1:'CG', 2:'GC', 3:'UA'}
        self.full_mapping = {0:'AU', 1:'UA', 2:'GC', 3:'CG', 4:'GU', 5:'UG'}
        self.reverse_action  = {0:3, 1:2, 2:1, 3:0}
        self.get_representations()

    def get_representations(self):
        """
        Generate the representations of the nucleotide sequence of required length based on the the target structure
        """
        self.hd = 100
        self.reward = 0
        k = self.kernel_size
        size, length = self.target.structure_encoding.shape
        self.str = ['-'] * length # List of chars ['-', 'A', "C", "G", "U"]

        self.nucleotide_encoding = np.zeros([4, length + k * 2])
        self.structure_encoding = np.zeros([size, length + k * 2])
        self.structure_encoding[:,k:-k] = self.target.structure_encoding
        
    @property
    def string(self):
        """
        Returns the nucleotide sequence of the solution in string format
        """
        return ''.join(self.str)
    
    def str_action(self, action):

        if self.target.seq[self.index] == '(':
            pair = self.target.paired_sites[self.index]
            self.str[pair] = self.mapping[action][1]
        else:
            self.str[self.index] = self.mapping[action][0]

    def matrix_action(self, action):
        """
        Set the nucleotide one-hot representation at the current step

        If at the current step the target structure contains an opening bracket, 
        pair the opposing closing bracket with an opposing nucleotide

        """
        ind1 = self.padded_index
        ind2 = self.index

        self.nucleotide_encoding[action, ind1] = 1

        if self.target.seq[ind2] == '(':
            pair = self.target.paired_sites[ind2] + self.kernel_size
            index = self.reverse_action[action] 
            self.nucleotide_encoding[index, pair] = 1 

    def find_next_unfilled(self):
        """
        Go to the next unfilled nucleotide
        """
        count = 1
        string = self.string[self.index:]
        while True:
            if count >= self.target.len - self.index - 1: break
            if string[count] == '-': break
            count += 1
        
        self.index += count
        self.padded_index += count

    def get_state(self, reshape=False):
        """
        Return the current state of the solution (window around the current nucleotide)
        """

        if self.use_padding:
            start = self.padded_index - self.kernel_size 
            end   = self.padded_index + self.kernel_size

            if self.config['use_nucleotides']:
                state = np.vstack([self.structure_encoding[:,start:end], self.nucleotide_encoding[:,start:end]])
            else:
                state = self.structure_encoding[:, start:end]
            state = np.expand_dims(state, axis=2)

        return state
    
    def evaluate(self, string=None, verbose=False, permute=False):
        """
        Evaluate the current solution, measure the hamming distance between the folded structure and the target
        """

        if string is None: string = self.string
        self.folded_design, self.fe = fold_fn(string)
        self.hd, mismatch_indices = hamming_distance(self.target.seq, self.folded_design)

        # Permutations
        if permute and 0 < self.hd <= self.config['permutation_threshold']:
            self.str, self.hd = self.local_improvement(mismatch_indices, budget=self.config['permutation_budget'], verbose=verbose)
            self.folded_design, self.fe = fold_fn(self.string)
            
        reward = (1 - float(self.hd)/float(self.target.len)) ** self.reward_exp
        gcau = self.gcau_content()
        if gcau['U'] < 0.12:
            reward = reward/2 + reward/2 * (0.12 - gcau['U'])

        self.reward = reward
        return reward, self.hd, mismatch_indices

    def local_improvement(self, original_mismatch_indices, budget=20, verbose=True):
        """
        Performs a local improvement on the mismatch sites 
        """
 
        budget = 

        def get_surrounding(mismatches):
            """
            Get indices of nucleotides around the mistmatches within a radius
            """
            window = range(1, self.config['permutation_radius']+1)
            all_indices = []
            for index in mismatches:
                for i in window:
                    if index - i >= 0:
                        all_indices.append(index-i)
                    if index + i <= self.target.len -1:
                        all_indices.append(index+i)
            mismatches = list(mismatches) + all_indices
            return set(all_indices)

        original_mismatch_indices = get_surrounding(original_mismatch_indices)
        best_permutation, best_mismatch_indices = None, None

        step = 0
        hd = self.hd
        min_hd = 500
        while self.hd != 0 and step < self.config['permutation_budget']:
            if verbose:
                print('Permutation #{:3}, HD: {:2}'.format(step, self.hd), end='\r')
            
            if min_hd > self.hd:
                permutation = copy.deepcopy(self.str)
                mismatch_indices = original_mismatch_indices
            else:
                permutation = copy.deepcopy(best_permutation)
                mismatch_indices = best_mismatch_indices

            for mismatch in mismatch_indices:

                mapping = self.mapping
                action = random.randint(0,3)

                if self.config['allow_gu_permutations']:
                    symbol = self.target.seq[mismatch]
                    if symbol == '(' or symbol == ')':
                        action = random.randint(0, 5)
                        mapping = self.full_mapping

                pair = None
                # In case of a stem find the paired nucleotide
                for key, item in self.target.paired_sites.items():
                    if key  == mismatch: pair = item
                    if item == mismatch: pair = key
                if random.random() > self.config['mutation_probability']:
                    permutation[mismatch] = mapping[action][0]
                    if pair is not None:
                        permutation[pair] = mapping[action][1]

            string = ''.join(permutation)
            _, hd, mismatch_indices = self.evaluate(string, permute=False)
            if hd < min_hd: 
                min_hd = hd
                best_mismatch_indices = get_surrounding(mismatch_indices)
                best_permutation = copy.deepcopy(permutation)

            step += 1
        if hd == 0 and verbose:
            print('\nPermutation succesful in {}/{} steps.'.format(step, budget))
        return best_permutation, min_hd

    def gcau_content(self):
        """
        Get the proportions of each nucleotide in the generated solution
        """
        gcau = dict(G=0, C=0, A=0, U=0)
        increment = 1. / len(self.str)
        for nucleotide in self.str:
            try: gcau[nucleotide] += increment
            except: pass

        return gcau

    def summary(self, print_output=False, colorize=True):
        """
        Formatted summary of the solution
        """
        gcau = self.gcau_content()
        date = datetime.datetime.now().strftime("%m-%d_%H-%M")
        separator = '\n\n' + ''.join(["*"] * 20) + ' ' + date + ' ' + ''.join(["*"] * 20)
        header = '\n\nSeq Nr. {:5}, Len: {:3}, DBr: {:.2f}, HD: {:3}, Reward: {:.5f}'.format(
            self.target.file_nr, 
            self.target.len, 
            self.target.db_ratio, 
            self.hd, 
            self.reward)

        header += '  ||   G:{:.2f} | C:{:.2f} | A:{:.2f} | U:{:.2f} |'.format(
            gcau['G'], 
            gcau['C'], 
            gcau['A'], 
            gcau['U'])

        solution = 'S:  ' + self.string
        folded   = 'F:  ' + self.folded_design
        target   = 'T:  ' + self.target.seq
        detail2  = 'D2; ' + self.target.markers
        free_en  = 'FE: {:.3f}'.format(self.fe)
        
        summary = [header, solution, folded, target, detail2, free_en]
        if colorize:
            colored  = 'S:  ' + colorize_nucleotides(self.string)
            mm1, mm2 = highlight_mismatches(self.folded_design, self.target.seq)
            hl1 = 'F:  ' + mm2
            hl2 = 'T:  ' + mm1
            console_output = [header, colored, hl1, hl2, free_en]
        else: 
            console_output = summary
        if print_output:
            for line in console_output:
                print(line)
        return summary

    def write_solution(self):
        """
        Writes a solution to the results.log file in the model folder
        """
        date = datetime.datetime.now().strftime("%m-%d_%H-%M")
        if 'path' in self.config.keys():
            path = os.path.join(self.config['path'], 'results.log')
        else:
            date = datetime.datetime.strftime
            path = os.path.join(settings.RESULTS, 'results__{}.log'.format(date))

        lines = self.summary()
        with open(os.path.join(path), 'a+') as f:
            for line in lines:
                f.write(line + '\n')

    def visualize(self, auto=False):
        """
        Call the forna container and visualize the dataset
        """
        driver = create_browser('display')
        
        show_rna(self.folded_design, self.string, driver=driver, html='display')
        print(self.summary())
        input()