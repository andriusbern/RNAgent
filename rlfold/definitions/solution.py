import numpy as np
import copy, random, os, datetime, sys
from rlfold.definitions import Sequence, hamming_distance
from rlfold.definitions import colorize_nucleotides, highlight_mismatches
from rlfold.utils import create_browser, show_rna
import rlfold.settings as settings
import forgi
import time as t
try:
    sys.path.remove('/usr/local/lib/python3.6/site-packages')
except:
    pass
import RNA

if settings.os == 'linux':
    # RNA.cvar.dangles = 2
    # RNA.cvar.noGU = 0

    param_files = {
        1: 'rna_turner2004.par',
        2: 'rna_turner1999.par',
        3: 'rna_andronescu2007.par',
        4: 'rna_langdon2018.par',
    }
    # params = os.path.join(settings.MAIN_DIR, 'utils', param_files[])
    # RNA.read_parameter_file(params)
    # RNA.cvar.uniq_ML = 1
    # RNA.cvar.no_closingGU = 1
    # RNA.cvar.betaScale = 1.5
    # RNA.cvar.temperature = 40.0
    fold_fn = RNA.fold
    
elif settings.os == 'win32':
    from .vienna import fold
    fold_fn = fold

class Solution(object):
    """
    Class for logging generated nucleotide sequence solutions
    """
    def __init__(self, target, config=None, string=None, time=None, source='rlfold'):
        self.config = config
        self.target = target
        self.time = None
        self.start = None
        self.mapping         = {0:'A', 1:'C', 2:'G', 3:'U'}
        self.reverse_mapping = {'A':0, 'C':1, 'G':2, 'U':3}
        self.reverse_action  = {0:3, 1:2, 2:1, 3:0}
        self.source = source
        self.mismatch_indices = None
        
        self.str = []
        self.use_full_state = self.config['full_state']
        if 'use_padding' in self.config.keys():
            self.use_padding = self.config['use_padding']
        else:
            self.use_padding = True

        # Statistics
        self.solution_id = settings.get_solution_id()
        self.hd = 100 # Hamming
        self.md = 0   # Mountain
        self.fe = 0   # Gibbs free energy
        self.probability = None

        # Partition function
        self.partition_prob = None
        self.partition_fn = None

        # Centroid
        self.centroid_structure = None
        self.centroid_dist = None
        self.centroid_en = None

        # MEA
        self.MEA_structure = None
        self.MEA = None
        self.MEA_en = None
        self.ensemble_diversity = None
        self.ensemble_defect = None

        self.reward = 0
        self.folded_design = ''
        self.reward_exp = config['reward_exp']
        self.kernel_size = config['kernel_size']
        self.index = 0
        self.padded_index = self.kernel_size
        self.get_representations()
        if string is not None:
            self.str = [x for x in string]
            self.evaluate(string, permute=False, compute_statistics=True)
        if time is None:
            self.start = t.time()
        else:
            self.start = None
            self.time = time

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

        start = self.padded_index - self.kernel_size 
        end   = self.padded_index + self.kernel_size

        if self.config['use_nucleotides']:
            state = np.vstack([self.structure_encoding[:,start:end], self.nucleotide_encoding[:,start:end]])
        else:
            state = self.structure_encoding[:, start:end]
        
        if reshape:
            state = state.flatten()
        else:
            state = np.expand_dims(state, axis=2)
        return state

    def compute_statistics(self):
        model_details = RNA.md()
        fc = RNA.fold_compound(self.string, model_details)
        fold, fe = fc.mfe()
        fc.exp_params_rescale(self.fe)
        self.partition_prob, self.partition_fn = fc.pf()
        self.centroid_structure, self.centroid_dist = fc.centroid()
        self.centroid_en = fc.eval_structure(self.centroid_structure)
        self.MEA_structure, self.MEA = fc.MEA()
        self.MEA_en = fc.eval_structure(self.MEA_structure)
        self.probability = fc.pr_structure(fold)
        self.ensemble_diversity = fc.mean_bp_distance()
        self.ensemble_defect = fc.ensemble_defect(fold)
        self.md = RNA.dist_mountain(self.target.seq, fold)
    
    def evaluate(self, string=None, verbose=False, permute=False, compute_statistics=False, boost=False):
        """
        Evaluate the current solution, measure the hamming distance between the folded structure and the target
        """
        if boost and string is None:
            self.boost()
        if string is None: string = self.string

        self.folded_design, self.fe = RNA.fold(string)
        # self.folded_design, self.fe = fc.mfe()
        self.hd, self.mismatch_indices = hamming_distance(self.target.seq, self.folded_design)

        if compute_statistics:
            self.compute_statistics()
        # Evaluate distance

        # Permutations
        if permute and 0 < self.hd <= self.config['permutation_threshold']:
            self.str, self.hd = self.local_improvement(self.mismatch_indices, budget=self.config['permutation_budget'], verbose=verbose)
            self.folded_design, self.fe = fold_fn(self.string)
            _, self.mismatch_indices = hamming_distance(self.target.seq, self.folded_design)
            # self.folded_design = Sequence(self.folded_design, encoding_type=self.config['encoding_type'])
            
        reward = (1 - float(self.hd)/float(self.target.len)) ** self.reward_exp
        gcau = self.gcau_content()
        if gcau['U'] < 0.12:
            reward = reward/2 + reward/2 * (0.12 - gcau['U'])

        self.reward = reward
        if self.start is not None:
            self.time = t.time() - self.start
        return reward, self.hd, self.mismatch_indices

    def local_improvement(self, original_mismatch_indices, budget=20, verbose=True):
        """
        Performs a local improvement on the mismatch sites 
        """
        step = 0
        hd = self.hd

        budget = self.config['permutation_budget']

        window = range(1, self.config['permutation_radius']+1)
        def get_surrounding(mismatches):
            all_indices = []
            for index in mismatches:
                for i in window:
                    if index - i >= 0:
                        all_indices.append(index-i)
                    if index + i <= self.target.len -1:
                        all_indices.append(index+i)
            mismatches = list(mismatches)
            mismatches += all_indices
            return set(all_indices)

        original_mismatch_indices = get_surrounding(original_mismatch_indices)

        best_permutation, best_mismatch_indices = None, None
        min_hd = 500
        p = 0
        while hd != 0 and step < budget:
            # if verbose:
            print('Permutation #{:3}, HD: {:2}'.format(step, hd), end='\r')
            
            if min_hd > self.hd:
                permutation = copy.deepcopy(self.str)
                mismatch_indices = original_mismatch_indices
            else:
                permutation = copy.deepcopy(best_permutation)
                mismatch_indices = best_mismatch_indices
            try:
                for mismatch in mismatch_indices:
                    if self.config['allow_gu_permutations']:
                        symbol = self.target.seq[mismatch]
                        if symbol == '.':
                            action = random.randint(0, 3)
                            mapping = {0:'A', 1:'C', 2:'G', 3:'U'}

                        elif symbol == '(' or symbol == ')':
                            action = random.randint(0, 5)
                            mapping = {0:'AU', 1:'UA', 2:'GC', 3:'CG', 4:'GU', 5:'UG'}
                    else:
                        mapping = {0:'AU', 1:'UA', 2:'GC', 3:'CG'}
                        action = random.randint(0,3)

                    pair = None
                    # In case of a stem find the paired nucleotide
                    for key, item in self.target.paired_sites.items():
                        if key  == mismatch: pair = item
                        if item == mismatch: pair = key
                    if random.random() > self.config['mutation_probability']:
                        permutation[mismatch] = mapping[action][0]
                        if pair is not None:
                            permutation[pair] = mapping[action][1]
            except:
                pass

            string = ''.join(permutation)
            _, hd, mismatch_indices = self.evaluate(string, permute=False)
            if hd < min_hd: 
                min_hd = hd
                best_mismatch_indices = get_surrounding(mismatch_indices)
                best_permutation = copy.deepcopy(permutation)

            step += 1
            # mm1, mm2 = highlight_mismatches(string, self.string)
            # print(mm1)
            # print(mm2)
            # input()
        if hd == 0:
            self.source = 'rlfold*'
            if verbose:
                print('\nPermutation successful in {}/{} steps.'.format(step, budget))
        return best_permutation, min_hd


    def boost(self, full=False):
        # Internal
        bg, = forgi.load_rna(self.target.seq)
        for i in bg.iloop_iterator():
            dims = bg.get_node_dimensions(i)
            indices = bg.elements_to_nucleotides([i])
            strand1 = indices[:dims[0]]
            strand2 = indices[-dims[1]:]

            if dims[1] > dims[0]:
                strand1, strand2 = strand2, strand1
            dims = (max(dims), min(dims))

            if dims == (1, 1):
                if random.random() > 0.5:
                    self.str[strand1[0]-1] = 'U' # G
                    self.str[strand2[-1]-1] = 'U' # A

            if dims == (2, 1):
                if random.random() > 0.5:
                    self.str[strand1[0]-1], self.str[strand1[1]-1] = 'U', 'C' # G, C / G, A
                    self.str[strand2[0]-1] = 'U' # A / G

            if full and dims == (2, 2):
                if random.random() > 0.5:
                    self.str[strand1[0]-1], self.str[strand1[1]-1] = 'U', 'G'
                    self.str[strand2[0]-1], self.str[strand2[1]-1] = 'U', 'G'


            if (dims[0] > 2 and dims[1] > 2) and (dims[0] >= 3 or dims[1] >= 3):
                if random.random() > 0.5:
                    self.str[strand1[0]-1], self.str[strand1[1]-1] = 'G', 'G'
                    self.str[strand2[0]-1], self.str[strand2[1]-1] = 'A', 'A'
    
        # Hairpins
        for h in bg.hloop_iterator():
            dims = bg.get_node_dimensions(h)
            indices = bg.elements_to_nucleotides([h])
            strand1 = indices[:dims[0]]
            if random.random() > 0.4 and len(indices) > 3:
                self.str[strand1[0]-1] = 'G'
                self.str[strand1[-1]-1] = 'A'
            
        # Close stems with GC/CG
        pair = ('C', 'G') if random.random() > 0.5 else ('G', 'C')
        for s in bg.stem_iterator():
            dims = bg.get_node_dimensions(s)
            indices = bg.elements_to_nucleotides([s])
            strand1 = indices[:dims[0]]
            strand2 = indices[-dims[1]:]

            # pair = ('C', 'G') if random.random() > 0.99 else ('G', 'C')
            if random.random() > 0.5:
                self.str[strand1[0]-1], self.str[strand2[-1]-1] = pair
                self.str[strand1[-1]-1], self.str[strand2[0]-1] = list(reversed(pair))
                
        pairs = ('C', 'A') if random.random() > 0.5 else ('A', 'C')
        if full:
            for m in list(bg.mloop_iterator())[1:-2]:
                dims = bg.get_node_dimensions(m)
                indices = bg.elements_to_nucleotides([m])
                strand1 = indices[:dims[0]]
                if len(strand1) > 1:
                    if random.random() > 0.5:
                        
                        self.str[strand1[0]-1], self.str[strand1[1]-1] = pairs

            # strand2 = indices[-dims[1]:]

        
    def variance_reward(self):
        """
        A reward that penalizes solutions that overuse or underuse nucleotide types.
        """

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