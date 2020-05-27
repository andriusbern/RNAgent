import numpy as np
import copy, random, os, datetime, sys, forgi
import time as t
from rlif.settings import ConfigManager as settings
from rlif.rna import DotBracket, hamming_distance
from rlif.rna import colorize_nucleotides, highlight_mismatches

fold_fn = settings.fold_fn
import RNA

class Solution(object):
    """
    Class for logging generated nucleotide sequence solutions
    """
    def __init__(self, target, config=None, string=None, time=None, source='rlif'):
        self.config = config
        self.target = target
        self.time   = time
        self.start  = None
        self.mapping        = {0:'AU', 1:'CG', 2:'GC', 3:'UA', 4:'GU', 5:'UG'}
        self.reverse_action = {0:3, 1:2, 2:1, 3:0, 4:5, 5:4}
        self.source = source
        self.str = ['-'] * target.len # List of chars ['-', 'A', "C", "G", "U"]

        # Statistics
        self.hd = 100 # Hamming distance
        self.md = 0   # Mountain
        self.fe = 0   # Gibbs free energy
        self.r  = 0   # Reward

        self.mismatch_indices = None
        self.folded_structure = ''
        self.reward_exp  = config['reward_exp']
        self.kernel_size = config['kernel_size']
        self.index = 0
        self.create_encoding()
        if self.config_check('boosting'):
            self.graph, = forgi.load_rna(self.target.seq)

        self.init_vars()
        if string is not None:
            self.str = [x for x in string]
            self.evaluate(string, permute=False, compute_statistics=True)

        if time is None:
            self.start = t.time()


    def config_check(self, parameter):
        """
        Checks whether the config is present in the .yml config file of the model
        Ensures backwards compatibility with updates
        """
        if self.config.get(parameter) is not None:
            return self.config[parameter]
        else:
            return False

    def add_constraint(self, constraint):
        pass

    def create_encoding(self):
        """
        Generate the representations of the nucleotide sequence of required length based on the the target structure
        """
        k = self.kernel_size
        self.rows, length = self.target.structure_encoding.shape
        structure_encoding = np.zeros([self.rows, length + k * 2]) # Padding
        structure_encoding[:,k:-k] = self.target.structure_encoding

        if self.config_check('use_nucleotides'):
            nucleotide_encoding = np.zeros([4, length + k * 2])
            self.encoding = np.vstack([structure_encoding, nucleotide_encoding])
        else:
            self.encoding = structure_encoding
        
    @property
    def string(self):
        """
        Returns the nucleotide sequence of the solution in string format
        """
        return ''.join(self.str)
    
    def insert_nucleotide(self, action):
        """
        Insert a nucleotide into the string based on an action
        """
        i = self.index
        k = self.kernel_size
        
        # Unpaired nucleotide
        self.encoding[self.rows + action, i + k] = 1
        self.str[i] = self.mapping[action][0]

        # Base pair
        if self.target.seq[i] == '(':
            try:
                pair_index = self.target.base_pair_indices[i]
                nucleotide = self.reverse_action[action]
                self.encoding[self.rows + nucleotide, pair_index + k] = 1
                self.str[pair_index] = self.mapping[action][1]
            except:
                pass

    def find_next_unfilled(self):
        """
        Set current index to the next unfilled nucleotide
        """
        index = self.index
        while index < self.target.len - 1:
            if self.str[index] == '-':
                break
            index += 1
        self.index = index
        
    def get_state(self, reshape=False):
        """
        Return the current state of the solution (window around the current nucleotide)
        """
        i = self.index 
        k = self.kernel_size
        state = self.encoding[:, i:i+2*k]
        return state.flatten() if reshape else np.expand_dims(state, axis=2)
    
    def evaluate(self, string=None, permute=False, compute_statistics=False, boost=False, reward=False, verbose=False):
        """
        Evaluate the current solution, measure the hamming distance between the folded structure and the target
        """
        if boost and string is None:
            self.boost()

        if string is None: string = self.string

        self.folded_structure, self.fe = fold_fn(string)
        self.hd, self.mismatch_indices = hamming_distance(self.target.seq, self.folded_structure)

        # Permutations
        if permute and 0 < self.hd <= self.config['permutation_threshold']:
            self.str, self.hd = self.permute(self.mismatch_indices, verbose=verbose)
            self.folded_structure, self.fe = fold_fn(self.string)
            self.hd, self.mismatch_indices = hamming_distance(self.target.seq, self.folded_structure)

        if compute_statistics: self.compute_statistics()

        if reward:
            self.r = (1 - float(self.hd)/float(self.target.len)) ** self.reward_exp
            gcau = self.gcau_content()
            if self.hd == 0:
                self.r += .25
            if self.config_check('gc_penalty') or self.config_check('gc'):
                self.r = self.r * 0.85 + np.clip(gcau['U'], 0, .15)

        if self.start is not None:
            self.time = t.time() - self.start
        return self.r, self.hd, self.mismatch_indices

    def permute(self, original_mismatch_indices, verbose=True):
        """
        Performs a local improvement on the mismatch sites 
        """
        budget = self.config['permutation_budget']
        radius = self.config['permutation_radius']
        window = range(1, radius+1)

        # Get indices of nucleotides around the mismatches within a radius
        def get_surrounding(mismatches): 
            all_indices = []
            probs = {}
            probabilities = list(np.linspace(0.01, self.config['mutation_probability'], radius+1))[:-1]
            for index in mismatches:
                for i in window:
                    probs[index] = self.config['mutation_probability']
                    # Nucleotides at i+1 and i-1
                    if index - i >= 0:
                        all_indices.append(index-i)
                        probs[index-i] = probabilities[i-1]
                    if index + i <= self.target.len -1:
                        all_indices.append(index+i)
                        probs[index+i] = probabilities[-i]
            mismatches = list(mismatches) + all_indices
            return set(all_indices), probs

        best_permutation, permutation = copy.deepcopy(self.str), copy.deepcopy(self.str)
        hd, min_hd = self.hd, self.hd
        best_mismatch_indices, probs = get_surrounding(original_mismatch_indices)
        step = 0
        while hd != 0 and step < budget:
            for mismatch in best_mismatch_indices:
                if random.random() < probs[mismatch]:
                    action = random.randint(0,3)
                    if self.config['allow_gu_permutations'] and self.target.seq[mismatch] != '.':
                        action = random.randint(0, 5)

                    # In case of a stem/helix find the paired nucleotide
                    permutation[mismatch] = self.mapping[action][0]
                    if self.target.seq[mismatch] != '.':
                        if self.target.base_pair_indices.get(mismatch) is not None:
                            pair = self.target.base_pair_indices.get(mismatch)
                        if self.target.rev_base_pair_indices.get(mismatch) is not None:
                            pair = self.target.rev_base_pair_indices.get(mismatch)
                        permutation[pair] = self.mapping[action][1]

            string = ''.join(permutation)
            folded, _ = fold_fn(string)
            hd, mismatch_indices = hamming_distance(self.target.seq, folded)
            if hd < min_hd: 
                min_hd = hd
                best_permutation = copy.deepcopy(permutation)
                best_mismatch_indices, probs = get_surrounding(mismatch_indices)
            else:
                permutation = copy.deepcopy(best_permutation)
            step += 1

            if hd == 0:
                self.source = 'rlif*'
                # print('\nPermutation successful in {}/{} steps.'.format(step, budget))
                break

        return best_permutation, min_hd

    def compute_statistics(self):
        """
        Compute various statistics about the secondary RNA structure
        """
        model_details = RNA.md()
        fc = RNA.fold_compound(self.string, model_details)
        fold, fe = fc.mfe()
        fc.exp_params_rescale(self.fe)
        self.partition_prob, self.partition_fn = fc.pf()
        self.positional_entropy = fc.positional_entropy()
        self.centroid_structure, self.centroid_dist = fc.centroid()
        self.centroid_en = fc.eval_structure(self.centroid_structure)
        self.MEA_structure, self.MEA = fc.MEA()
        self.MEA_en = fc.eval_structure(self.MEA_structure)
        self.probability = fc.pr_structure(fold)
        self.ensemble_diversity = fc.mean_bp_distance()
        self.ensemble_defect = fc.ensemble_defect(fold)
        self.md = RNA.dist_mountain(self.target.seq, fold)

    def init_vars(self):
        self.probability = None
        self.positional_entropy = None
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

    def boost(self, full=False):
        # Internal
        
        for i in self.graph.iloop_iterator():
            dims = self.graph.get_node_dimensions(i)
            indices = self.graph.elements_to_nucleotides([i])
            strand1 = indices[:dims[0]]
            strand2 = indices[-dims[1]:]

            if dims[1] > dims[0]:
                strand1, strand2 = strand2, strand1
            dims = (max(dims), min(dims))

            if dims == (1, 1):
                pair = random.sample([('U', 'U'), ('G', 'A'), ('A', 'G')], 1)[0]
                if random.random() > 0.5:
                    self.str[strand1[0]-1], self.str[strand2[-1]-1] = pair
                    
            if dims == (2, 1):
                pair = random.sample([['U', 'C', 'U'], ['G', 'A', 'A']], 1)[0]
                if random.random() > 0.5:
                    self.str[strand1[0]-1], self.str[strand1[1]-1] = pair[:2] # G, C / G, A
                    self.str[strand2[0]-1] = pair[2] # A / G

            if full and dims == (2, 2):
                pair = random.sample([('G', 'U'), ('U', 'G')], 1)[0]
                if random.random() > 0.5:
                    self.str[strand1[0]-1], self.str[strand1[1]-1] = pair
                    self.str[strand2[0]-1], self.str[strand2[1]-1] = pair

            if (dims[0] > 2 and dims[1] > 2) and (dims[0] >= 3 or dims[1] >= 3):
                pair = random.sample([['G', 'A'],['A', 'G']], 1)[0]
                if random.random() > 0.5:
                    self.str[strand1[0]-1], self.str[strand1[1]-1] = pair
                    self.str[strand2[1]-1], self.str[strand2[0]-1] = pair
    
        # Hairpins
        for h in self.graph.hloop_iterator():
            indices = self.graph.elements_to_nucleotides([h])
            if random.random() > 0.5 and len(indices) > 3:
                # pair = random.sample([('U', 'U'), ('G', 'A')], 1)[0]
                pair = ['G', 'A']
                dims = self.graph.get_node_dimensions(h)
                strand1 = indices[:dims[0]]
                self.str[strand1[0]-1], self.str[strand1[-1]-1] = pair
            
        # # Close stems with GC/CG
        # for s in self.graph.stem_iterator():
        #     nucls = [('G', 'C'), ('C', 'G')] #('G', 'U'), ('U', 'G'), 
        #     if random.random() > 0.5:
        #         dims = self.graph.get_node_dimensions(s)
        #         indices = self.graph.elements_to_nucleotides([s])
        #         strand1 = indices[:dims[0]]
        #         strand2 = indices[-dims[1]:]
        #         pair = random.sample(nucls, 1)[0]
        #         print(pair)
        #         self.str[strand1[0]-1], self.str[strand2[-1]-1] = pair
        #         self.str[strand1[-1]-1], self.str[strand2[0]-1] = reversed(list(pair))

    def summary(self, print_output=False, colorize=True):
        """
        Formatted summary of the solution
        """
        gcau = self.gcau_content()
        date = datetime.datetime.now().strftime("%m-%d_%H-%M")
        separator = '\n\n' + ''.join(["*"] * 20) + ' ' + date + ' ' + ''.join(["*"] * 20)
        header = '\n\nSeq Nr. {:5}, Len: {:3}, DBr: {:.2f}, HD: {:3}, Reward: {:.5f}'.format(
                  self.target.file_nr, self.target.len, self.target.percent_unpaired, self.hd, self.r)
        header += '  ||   G:{:.2f} | C:{:.2f} | A:{:.2f} | U:{:.2f} |'.format(gcau['G'], gcau['C'], gcau['A'], gcau['U'])
        solution = 'S:  ' + self.string
        folded   = 'F:  ' + self.folded_structure
        target   = 'T:  ' + self.target.seq
        detail2  = 'D2; ' + self.target.struct_motifs
        free_en  = 'FE: {:.3f}'.format(self.fe)
        summary = [header, solution, folded, target, detail2, free_en]
        if colorize:
            colored  = 'S:  ' + colorize_nucleotides(self.string)
            mm1, mm2 = highlight_mismatches(self.folded_structure, self.target.seq)
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
        print("\n    %s\nMFE %s (%6.2f)" % (string, self.folded_structure, self.fe))
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
        
        show_rna(self.folded_structure, self.string, driver=driver, html='display')
        print(self.summary())
        input()

    
    # def md_action(self, action):
    #     """
    #     Multi-discrete action
    #     """
    #     ind1 = self.index + self.kernel_size
    #     ind2 = self.index
    #     self.nucleotide_encoding[ind1] = action + 1
    #     if self.target.seq[ind2] == '(':
    #         pair = self.target.base_pair_indices[ind2] + self.kernel_size
    #         self.nucleotide_encoding[pair] = self.reverse_action[action] + 1
                
    # def multistrand_action(self, action):
    #     """
    #     Multi-strand action
    #     """
    #     ind1 = self.ms_index
    #     ind2 = self.index
    #     ms_index2 = np.where(self.target.strand2==self.index+1)[0]

    #     self.nucleotide_encoding[0, ind1] = action + 1
    #     self.nucleotide_encoding[1, ms_index2] = action + 1

    #     if self.target.seq[ind2] == '(':
    #         try:
                
    #             pair = self.target.base_pair_indices[ind2]
    #             ms_indexp = np.where(self.target.strand1==pair+1)
    #             ms_index2p = np.where(self.target.strand2==pair+1)

    #             self.nucleotide_encoding[0, ms_indexp]  = self.reverse_action[action] + 1
    #             self.nucleotide_encoding[1, ms_index2p] = self.reverse_action[action] + 1
    #         except:

    #             pass
            # size, length = shape
        # if len(shape) > 1:
        #     size, length = shape
        # else:
        #     length = shape[0]

        # if self.config_check('multi_discrete'):
        #     self.nucleotide_encoding = np.zeros([length + k * 2])
        #     self.structure_encoding = np.zeros([length + k * 2])
        #     self.structure_encoding[k:-k] = self.target.structure_encoding
    #     # else:
    #         def str_action(self, action):
    #     """
    #     Insert a nucleotide into the string based on an action
    #     """
    #     self.str[self.index] = self.mapping[action][0]
    #     if self.target.seq[self.index] == '(':
    #         pair_index = self.target.base_pair_indices[self.index]
    #         self.str[pair_index] = self.mapping[action][1]

    # def matrix_action(self, action):
    #     """
    #     Set the nucleotide one-hot representation at the current step

    #     If at the current step the target structure contains an opening bracket, 
    #     pair the opposing closing bracket with an opposing nucleotide
    #     """
    #     ind1 = self.index + self.kernel_size
    #     ind2 = self.index

    #     self.nucleotide_encoding[action, ind1] = 1
    #     if self.target.seq[ind2] == '(':
    #         pair = self.target.base_pair_indices[ind2] + self.kernel_size
    #         index = self.reverse_action[action] 
    #         self.nucleotide_encoding[index, pair] = 1

    #  if self.config_check('multi_discrete'):
    #         state = np.hstack([self.structure_encoding[start:end], self.nucleotide_encoding[start:end]])