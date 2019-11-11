import numpy as np
import networkx as nx
import random, datetime, os, copy

from rlif.rna import DotBracket
from rlif.rna import hamming_distance, highlight_mismatches, colorize_nucleotides
from rlif.settings import ConfigManager as settings

if settings.os == 'linux':
    import RNA
    fold_fn = RNA.fold
elif settings.os == 'win32':
    from .vienna import fold
    fold_fn = fold


class GraphSolution(object):
    """
    A graph-based solution
    """
    def __init__(self, target, config):

        self.config = config
        self.target = target

        self.nucleotide_list = None
        self.index = 0
        self.hamming_distance = 100
        self.free_energy = 0
        self.reward = 0
        self.folded_structure = ''
        self.hd = 0

        self.current_graph_element = None
        self.current_subgraph = None
        self.subgraph_features = None

        # self.graph_mapping   = {'-': 0, 'U':0, 'A':1, 'C':2, 'G':3}
        # self.mapping = {'A':-2, 'U':-1, 'G':1, 'C':2}
        self.reverse_mapping = {0: 'A', 1: 'U', 2: 'G', 3: 'C'}
        self.reverse_action  = {0:1, 1:0, 2:3, 3:2}
        self.get_representations()
        self.get_new_subgraph()
        self.get_features()
        
    @property
    def string(self):
        """
        Returns the nucleotide sequence of the solution in string format
        """
        return ''.join(self.nucleotide_list)

    def str_action(self, action):

        # if current_nucleotide == '-':
        self.nucleotide_list[self.index] = self.reverse_mapping[action]
        if self.target.seq[self.index] == '(':
            try:
                pair = self.target.base_pair_indices[self.index]
            except:
                pair = 0
            self.nucleotide_list[pair] = self.reverse_mapping[self.reverse_action[action]]

    def graph_action(self, action):
        """
        Change the node features based on the new action

        Check if ints are needed for features
        """
        self.subgraph_features[self.index+1] == self.reverse_mapping[action]
        if self.target.seq[self.index] == '(':
        
            try:
                pair = self.target.base_pair_indices[self.index]
            except:
                pair = 0
            if pair+1 in self.current_subgraph.nodes:
                self.subgraph_features[pair+1] = self.reverse_mapping[self.reverse_action[action]]

    def find_next_unfilled(self):
        """
        Go to the next unfilled nucleotide
        """
        count = 1
        string = self.nucleotide_list[self.index:]
        while True:
            if count >= self.target.len - self.index - 1:
                break
            if string[count] == '-':
                break
            count += 1
        
        self.index += count

    def get_representations(self):
        _, length = self.target.structure_encoding.shape
        self.nucleotide_list = ['-'] * length # List of chars ['-', 'A', "C", "G", "U"]

    def get_new_subgraph(self):
        """
        If necessary change the current subgraph to next include next high level graph elements
        """
        element = self.target.graph.get_elem(self.index+1)
        if element != self.current_graph_element:
            self.current_subgraph = self.target.get_subgraph(self.index+1)
            self.current_graph_element = element

    def get_features(self):
        """
        Get the subgraph of the nucleotides surrounding the current index and annotate the nodes with 
        the nucleotide sequence generated so far.

        Current nucleotide is denoted as group -1, others are mapped using self.graph_mapping
        """

        # subgraph = self.target.get_subgraph(self.index)

        nucleotides = []
        for node in self.current_subgraph.nodes:
            num = node - 1
            current = self.nucleotide_list[num]
            if num == self.index:
                nucleotides.append('O')
            else:
                nucleotides.append(current)
        
        self.subgraph_features = dict(zip(self.current_subgraph.nodes, nucleotides))

        return self.subgraph_features
    
    def get_embedding(self):
        """
        """

    def evaluate(self, string=None, verbose=False, permute=False):
        """
        Evaluate the current solution, measure the hamming distance between the folded structure and the target
        """

        if string is None: string = self.string
        self.folded_structure, self.fe = fold_fn(string)
        # self.folded_structure = DotBracket(self.folded_structure, encoding_type=self.config['encoding_type'])
        # if self.config['detailed_comparison']:
        #     self.hd, mismatch_indices = hamming_distance(self.target.struct_motifs, self.folded_structure.struct_motifs)
        # else:
        self.hd, mismatch_indices = hamming_distance(self.target.seq, self.folded_structure)

        if permute and self.hd <= 5 and self.hd > 0:
            self.nucleotide_list, self.hd = self.local_improvement(mismatch_indices)
            self.folded_structure, self.fe = fold_fn(self.string)
            # self.folded_structure = DotBracket(self.folded_structure, encoding_type=self.config['encoding_type'])
            
        reward = (1 - float(self.hd)/float(self.target.len)) ** self.config['reward_exp']
        # gcau = self.gcau_content()
        # if gcau['U'] < 0.12:
        #     reward = reward/2 + reward/2 * (0.12 - gcau['U'])
        if verbose:
            print('\nFolded sequence : \n {} \n Target: \n   {}'.format(self.folded_structure, self.target.seq))
            print('\nHamming distance: {}\n'.format(reward))
        self.reward = reward
        return reward, self.hd

    def local_improvement(self, mismatch_indices, budget=20, surroundings=True):
        """
        Performs a local improvement on the mismatch sites 
        """
        # reward = self.reward
        step = 0

        if surroundings:
            all_indices = []
            window = [1]
            for index in mismatch_indices:
                for i in window:
                    if index - i >= 0:
                        all_indices.append(index-i)
                    if index + i <= self.target.len -1:
                        all_indices.append(index+i)
            mismatch_indices = list(mismatch_indices)
            mismatch_indices += all_indices
        else:
            all_indices = mismatch_indices
        min_hd = 100
        while self.hd != 0 and step < budget:
            print('Permutation #{:3}, HD: {:2}'.format(step, self.hd), end='\r')
            permutation = copy.deepcopy(self.nucleotide_list)
            
            for mismatch in all_indices:
                pair = None
                for key, item in self.target.base_pair_indices.items():
                    if key  == mismatch: pair = item
                    if item == mismatch: pair = key

                action = random.randint(0, 3)
                permutation[mismatch] = self.reverse_mapping[action]
                if pair is not None:
                    permutation[pair] = self.reverse_mapping[self.reverse_action[action]]

            string = ''.join(permutation)
            _, hd = self.evaluate(string, permute=False)
            if hd < min_hd: min_hd = hd

            step += 1
            # mm1, mm2 = highlight_mismatches(string, self.string)
            # print(mm1)
            # print(mm2)
            # input()
        if hd == 0:
            print('\nPermutation succesful in {}/{} steps.'.format(step, budget))
        return permutation, min_hd


    def summary(self, print_output=False, colorize=True):
            """
            Formatted summary of the solution
            """
            gcau = self.gcau_content()
            date = datetime.datetime.now().strftime("%m-%d_%H-%M")
            separator = '\n\n' + ''.join(["*"] * 20) + ' ' + date + ' ' + ''.join(["*"] * 20)
            header = '\n\nSeq Nr. {:5}, Len: {:3}, DBr: {:.2f}, HD: {:3}, Reward: {:.5f}'.format(
                    self.target.file_nr, self.target.len, self.target.percent_unpaired, self.hd, self.reward)
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
                # mm3, mm4 = highlight_mismatches(self.target.struct_motifs, self.target.struct_motifs)
                hl1 = 'F:  ' + mm2
                hl2 = 'T:  ' + mm1
                console_output = [header, colored, hl1, hl4, free_en]
            else: 
                console_output = summary
            if print_output:
                for line in console_output:
                    print(line)
            return summary

    def gcau_content(self):
        """
        Get the proportions of each nucleotide in the generated solution
        """
        gcau = dict(G=0, C=0, A=0, U=0)
        increment = 1. / len(self.nucleotide_list)
        for nucleotide in self.nucleotide_list:
            gcau[nucleotide] += increment

        return gcau


        # size, length = self.target.graph_based_encoding
        # self.nucleotide_encoding = np.zeros([2, length + k * 2])
        # self.structure_encoding = np.zeros([1, length + k * 2])
        # self.structure_encoding[:, k:-k] = self.target.graph_based_encoding[0, :]
