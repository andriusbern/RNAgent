import numpy as np
import copy, random, os, datetime
from rlfold.definitions import Sequence, hamming_distance
from rlfold.definitions import colorize_nucleotides, highlight_mismatches
from rlfold.utils import create_browser, show_rna
import rlfold.settings as settings
import time as t
from rlfold.definitions import Solution

# if settings.os == 'linux':
#     import RNA
#     RNA.cvar.dangles = 1
#     RNA.cvar.noGU = 0

#     param_files = {
#         1: 'rna_turner2004.par',
#         2: 'rna_turner1999.par',
#         3: 'rna_andronescu2007.par',
#         4: 'rna_langdon2018.par',
#     }
#     params = os.path.join(settings.MAIN_DIR, 'utils', param_files[2])
#     RNA.read_parameter_file(params)
#     # RNA.cvar.uniq_ML = 1
#     # RNA.cvar.no_closingGU = 1
#     # RNA.cvar.betaScale = 1.5
#     # RNA.cvar.temperature = 40.0
#     fold_fn = RNA.fold
    
elif settings.os == 'win32':
    from .vienna import fold
    fold_fn = fold

class Spliceable(Solution):
    """
    Input for the splicer agent
    """
    def __init__(self, target, config, mismatches, string=None, time=None, source='rlfold'):
        Solution.__init__(self, target, config, string=None, time=None, source='rlfold')

    def get_representations(self):
        """
        Generate the representations of the nucleotide sequence of required length based on the the target structure
        """

        self.hd = 100
        self.reward = 0
        k = self.kernel_size
        size, length = self.target.structure_encoding.shape
        self.str = ['-'] * length # List of chars ['-', 'A', "C", "G", "U"]

        if self.use_padding:
            self.nucleotide_encoding = np.zeros([4, length + k * 2])
            # Padded target sequence
            self.structure_encoding = np.zeros([size, length + k * 2])
            self.structure_encoding[:,k:-k] = self.target.structure_encoding

            # Padded mismatches
            # Encoded as:
                # First row : matches 1
                # Second row : mismatches 1
            self.mismatch_encoding = np.zeros([2, size, length + k * 2])
        else:
            self.nucleotide_encoding = np.zeros([4, self.target.len]) # One-hot representation of the generated nucleotide sequence
        
    @property
    def string(self):
        """
        Returns the nucleotide sequence of the solution in string format
        """
        return ''.join(self.str)
    
    @property
    def matrix(self):
        """
        Returns the matrix representation of the current solution without padding
        """
        if self.use_padding:
            return self.nucleotide_encoding[:, self.kernel_size:-self.kernel_size]
        else:
            return self.nucleotide_encoding

    def str_action(self, action):
        if self.str[self.index] == '-':
            self.str[self.index] = self.mapping[action]
        if self.target.seq[self.index] == '(':
            try:
                pair = self.target.paired_sites[self.index]
            except:
                pair = 0
            self.str[pair] = self.mapping[self.reverse_action[action]]

    def modification_action(self, action):
        pass
        
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
            try:
                pair = self.target.paired_sites[ind2] + self.kernel_size
            except:
                pair = 0
            index = self.reverse_action[action] 
            self.nucleotide_encoding[index, pair] = 1

    def find_next_unfilled(self):
        """
        Go to the next unfilled nucleotide
        """
        count = 1
        string = self.string[self.index:]
        while True:
            if count >= self.target.len - self.index - 1:
                break
            if string[count] == '-':
                break
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
                state = np.vstack(
                    [self.structure_encoding[:,start:end], 
                     self.nucleotide_encoding[:,start:end],
                     self.mismatches[start:end]])
            else:
                state = self.structure_encoding[:, start:end]
            state = np.expand_dims(state, axis=2)

        return state

    def compute_statistics(self):
        """
        Compute the ensemble statistics
        """
        fc = RNA.fold_compound(self.string)
        self.folded_design, self.fe = fc.mfe()
        self.partition_prob, self.partition_fn = fc.pf()
        self.centroid_structure, self.centroid_dist = fc.centroid()
        self.centroid_en = fc.eval_structure(self.centroid_structure)
        self.MEA_structure, self.MEA = fc.MEA()
        self.MEA_en = fc.eval_structure(self.MEA_structure)
        self.probability = fc.pr_structure(self.folded_design)
        self.ensemble_diversity = fc.mean_bp_distance()
        self.ensemble_defect = fc.ensemble_defect(self.folded_design)
    
    def evaluate(self, string=None, verbose=False, permute=False, compute_statistics=False):
        """
        Evaluate the current solution, measure the hamming distance between the folded structure and the target
        """

        if string is None: string = self.string
        fc = RNA.fold_compound(string)
        self.folded_design, self.fe = fc.mfe()

        if compute_statistics:
            self.compute_statistics()
        # Evaluate distance
        self.hd, mismatch_indices = hamming_distance(self.target.seq, self.folded_design)

        # Permutations
        if permute and 0 < self.hd <= self.config['permutation_threshold']:
            self.str, self.hd = self.local_improvement(mismatch_indices, budget=self.config['permutation_budget'], verbose=verbose)
            self.folded_design, self.fe = fold_fn(self.string)
            # self.folded_design = Sequence(self.folded_design, encoding_type=self.config['encoding_type'])
            
        reward = (1 - float(self.hd)/float(self.target.len)) ** self.reward_exp
        gcau = self.gcau_content()
        if gcau['U'] < 0.12:
            reward = reward/2 + reward/2 * (0.12 - gcau['U'])

        self.reward = reward
        if self.start is not None:
            self.time = t.time() - self.start
        return reward, self.hd, mismatch_indices

    def gcau_content(self):
        """
        Get the proportions of each nucleotide in the generated solution
        """
        gcau = dict(G=0, C=0, A=0, U=0)
        increment = 1. / len(self.str)
        for nucleotide in self.str:
            try:
                gcau[nucleotide] += increment
            except:
                pass

        return gcau

    def summary(self, print_output=False, colorize=True):
        """
        Formatted summary of the solution
        """
        gcau = self.gcau_content()
        date = datetime.datetime.now().strftime("%m-%d_%H-%M")
        separator = '\n\n' + ''.join(["*"] * 20) + ' ' + date + ' ' + ''.join(["*"] * 20)
        header = '\n\nSeq Nr. {:5}, Len: {:3}, DBr: {:.2f}, HD: {:3}, Reward: {:.5f}'.format(
                  self.target.file_nr, self.target.len, self.target.db_ratio, self.hd, self.reward)
        header += '  ||   G:{:.2f} | C:{:.2f} | A:{:.2f} | U:{:.2f} |'.format(gcau['G'], gcau['C'], gcau['A'], gcau['U'])
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

    def short_summary(self):
        print("\n    %s\nMFE %s (%6.2f)" % (string, self.folded_design, self.fe))
        print("PF  %s [%6.2f]" % (self.partition_prob, self.partition_fn))
        print("CNT %s {%6.2f d=%.2f}" % (self.centroid_structure, self.centroid_en, self.centroid_dist))
        print("MEA %s {%6.2f MEA=%.2f}" % (self.MEA_structure, self.MEA_en, self.MEA))
        print(" frequency of mfe structure in ensemble %g; ensemble diversity %-6.2f" % (self.probability), self.ensemble_diversity)

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
        # print(self.target.seq)
        # print(seq.markers)
        input()
        # if not auto: input()
        # else: time.sleep(2)