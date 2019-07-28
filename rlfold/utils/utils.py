import numpy as np
import rlfold.settings as settings
import os, random
import matplotlib.pyplot as plt
import copy, datetime, yaml

if settings.os == 'linux':
    import RNA
    fold_fn = RNA.fold
elif settings.os == 'win32':
    from .vienna import fold
    fold_fn = fold

# plt.ion()

def load_sequence(num, dataset='rfam_learn_train'):
    print('Loading sequence #{}...'.format(num), end='\r')
    path = os.path.join(settings.DATA, dataset)
    filename = os.path.join(path, '{}.rna'.format(num))
    # try:
    with open(filename, 'r') as f:
        seq = f.readline()[:-1]
    return Sequence(seq, filename, num)
    # except:
    #     return []
    
def load_length_metadata(dataset, length):
    filename = os.path.join(settings.DATA, 'metadata', dataset, 'len', '{}.yml'.format(length))
    print('Loaded metadata from {}'.format(filename))
    try:
        with open(filename, 'r') as f:
            data = yaml.load(f)
        return data
    except:
        return []

def colorize_nucleotides(sequence):
    """
    Color coding for the console output for nucleotide sequences
    """
    mapping = {'A':'\033[4;30;42mA\033[0m',
               'U':'\033[4;30;46mU\033[0m',
               'G':'\033[4;30;41mG\033[0m',
               'C':'\033[4;30;45mC\033[0m',
               '-':'\033[4;37;40m-\033[0m'}    
    return ''.join([mapping[x] for x in sequence])

def highlight_mismatches(seq1, seq2):
    """
    Color highlighting of mismatches between strings
    """
    mseq1, mseq2 = '', '' # Modified sequences
    for char1, char2 in zip(seq1, seq2):
        if char1 != char2: # Mismatch
            mseq1 += '\033[2;31;40m{}\033[0m'.format(char1)
            mseq2 += '\033[2;31;40m{}\033[0m'.format(char2)
        else:              # Match
            mseq1 += '\033[2;32;40m{}\033[0m'.format(char1)
            mseq2 += '\033[2;32;40m{}\033[0m'.format(char2)

    return mseq1, mseq2

class Dataset(object):
    def __init__(self, length=None, start=1, n_seqs=65000, sequences=None, dataset='rfam_learn_train'):
        print('Loading dataset: {}, sequences: {}...\n'.format(dataset, n_seqs))
        self.dataset = dataset
        self.path = os.path.join(settings.DATA, dataset)
        self.sequence_length = length

        if length is not None:
            if isinstance(length, int):
                length_indices = load_length_metadata(dataset, length)
                self.sequences = [load_sequence(x) for x in length_indices[:n_seqs]]
            if isinstance(length, list):
                self.sequences = []
                print(length)
                for l in length:
                    print('\n\n', l)
                    length_indices = load_length_metadata(dataset, l)    
                    self.sequences += [load_sequence(x) for x in length_indices[:n_seqs]]

        if sequences is not None:
            self.sequences = sequences
        elif sequences is None and length is None:
            self.sequences = [load_sequence(x) for x in range(start, start+n_seqs)]
            self.sequences = [x for x in self.sequences if x is not None]
        self.n_seqs = len(self.sequences)
        self.current_index = -1
        self.statistics()

    def __repr__(self):
        return self.statistics()

    def __getitem__(self, index):
        return self.sequences[index]

    def visualize(self, structure):
        """
        Call the forna server
        """
        pass
    
    def statistics(self):
        avg_length = np.mean([seq.len for seq in self.sequences])
        msg = 'Sequences: {}\n'.format(len(self.sequences))+ \
              'Average length: {}\n'.format(avg_length)
        return msg
    
    def length_distribution(self, bins=50):
        lengths = [seq.len for seq in self.sequences]
        plt.hist(lengths, bins=bins, histtype='bar', ec='black', alpha=0.5, color='r')
        plt.ylabel('Number of sequences')
        plt.xlabel('Sequence length')
        plt.show()

    def show_all(self):
        
        for i in range(self.n_seqs):
            self.sequences[0]
            plt.cla()
            plt.imshow(seq)
            plt.show()
            plt.pause(.0001)

    def find_short(self):
        """
        """
        shorties = [x for x in range(self.n_seqs) if self.sequences[x].len < 55]
        print(shorties)

    def write_csv(self):
        """
        Write the detailed statistics about every sequence into a .csv file
        """

    def length_grouping(self):
        lengths = {}
        for seq in self.sequences:
            if seq.len not in lengths.keys(): lengths[seq.len] = []
            lengths[seq.len].append(seq.file_nr)

        for length in lengths.keys():
            dirname = os.path.join(settings.DATA, 'metadata', self.dataset, 'len')
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            filename = os.path.join(dirname, '{}.yml'.format(length))
            with open(filename, 'w') as f:
                yaml.dump(lengths[length], f)
        return lengths


class Sequence(object):
    """
    Dot-bracket notation sequence
    """
    def __init__(self, sequence, file_id=None, file_nr=None):
        self.seq = sequence
        self.len = len(sequence)
        self.bin = self.to_binary()
        self.file_id = file_id
        self.file_nr = file_nr
        self.db_ratio = float(sum([1 if x == '.' else 0 for x in self.seq])) / self.len
        self.loops = self.find_loops()
        self.n_loops = len(self.loops)
        self.paired_sites = self.find_complementary()
        self.mat = self.to_matrix()
        
    def __repr__(self):
        return self.seq

    def to_binary(self):
        """
        Convert 
        """
        mapping = {'.': 0, '(':1, ')':1}
        binary = [mapping[self.seq[x]] for x in range(len(self.seq))]
        return np.array(binary)

    def to_matrix(self):
        """
        Convert the current sequence into a matrix format:

        ..((..))..  ---.
        1100110011     |
        0011000000 <---`
        0000001100
        """
        mapping = {'.': 0, '(':1, ')':1}
        matrix = np.zeros([2, self.len])
        for i in range(self.len):
            matrix[mapping[self.seq[i]], i] = 1

        return matrix
    
    def find_bracket_pairings(self):
        """
        Fi
        """
        site_pairings = {}
        for i in range(self.len):
            if self.seq[i] == '(':
                opening, closing = True, True
                f_count, r_count = 0, 0
                for j in range(i+1, self.len):
                    if self.seq[j] == '(':
                        if opening:
                            f_count += 1
                    elif self.seq[j] == ')':
                        opening = False
                        r_count += 1
                    if f_count == r_count:
                        index = np.clip(j+1, 0, self.len - 1)
                        site_pairings[i] = index

        return site_pairings

    def find_loops(self):
        """
        Find the number of loops in the sequence
        """
        loops = []
        opening = False
        for i in range(self.len-1):
            if self.seq[i] == '(':
                opening = True
                last_opening = i
            if opening:
                if self.seq[i] == ')':
                    loops.append([last_opening, i])
                    opening = False

        return loops        
                
    def find_complementary(self):
        """
        Find complementary nucleotide indices by expanding outwards
        from loops
        """
        seq = [x for x in self.seq]
        pairs = {}
        for loop in self.loops:
            stop = False
            i1, i2 = loop[0], loop[1]
            # Add pair
            if seq[i1] == '(' and seq[i2] == ')':
                pairs[i1] = i2
                seq[i1], seq[i2] = '-', '-'
                i1 -= 1
                i2 += 1
            while True:
                # Expansion
                if seq[i1] != '(':
                    i1 -= 1
                if seq[i2] != ')':
                    i2 += 1
                # Bounds
                if i2 > self.len - 1: i2 = self.len - 1
                if i1 <= 0: i1 = 0
                # Termination conditions
                if i1 < 0 or seq[i1] == ')': stop = True
                if i2 == self.len-1 or seq[i2] == '(': stop = True

                # Add pair
                if seq[i1] == '(' and seq[i2] == ')':
                    pairs[i1] = i2
                    seq[i1], seq[i2] = '-', '-'
                    i1 -= 1
                    i2 += 1

                if stop:
                    break

        return pairs
        
class Solution(object):
    """
    Class for logging generated nucleotide sequence solutions
    """
    def __init__(self, target, config, dataset=None):

        self.config = config
        self.mapping = {0:'A', 1:'C', 2:'G', 3:'U'}
        self.reverse_mapping = {'A':0, 'C':1, 'G':2, 'U':3}
        self.reverse_action = {0:3, 1:2, 2:1, 3:1}

        self.target = target
        self.folded = None
        self.str = None
        self.use_full_state = self.config['full_state']
        if 'use_padding' in self.config.keys():
            self.use_padding = self.config['use_padding']
        else:
            self.use_padding = True

        # Statistics
        self.solution_id = settings.get_solution_id()
        self.hd = 100
        self.fe = 0
        self.r = 0
        self.folded_design = ''
        self.reward_exp = config['reward_exp']
        self.kernel_size = config['kernel_size']
        self.current_nucleotide = 0
        self.padded_index = self.kernel_size
        self.get_representations()
    
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
            return self.mat[:, self.kernel_size:-self.kernel_size]
        else:
            return self.mat

    def get_representations(self):
        """
        Generate the representations of the nucleotide sequence of required length based on the the target structure
        """
        if self.use_padding:
            k = self.kernel_size
            size, length = self.target.mat.shape
            self.mat = np.zeros([4, length + k * 2])
            self.str = ['-'] * length
            # Padded target sequence
            self.ptarget = np.zeros([size, length + k * 2])
            self.ptarget[:,k:-k] = self.target.mat
        else:
            self.mat = np.zeros([4, self.target.len]) # One-hot representation of the generated nucleotide sequence
            self.bin = np.zeros([1, self.target.len]).squeeze() # A numerical representation of the nucleotide sequence [0, 1, 3, 2, 4...]
            self.str = ['-'] * self.target.len # List of chars ['-', 'A', "C", "G", "U"]


    def binary_action(self, action):
        """
        Append the binary string 
        """
        self.bin[self.current_nucleotide] = action
        if self.target.seq[self.current_nucleotide] == '(':
            pair = self.target.paired_sites[self.current_nucleotide]
            self.bin[pair] = self.reverse_action[action]

    def str_action(self, action):
        if self.str[self.current_nucleotide] == '-':
            self.str[self.current_nucleotide] = self.mapping[action]
        if self.target.seq[self.current_nucleotide] == '(':
            pair = self.target.paired_sites[self.current_nucleotide]
            self.str[pair] = self.mapping[self.reverse_action[action]]
        

    def matrix_action(self, action):
        """
        Set the nucleotide one-hot representation at the current step

        If at the current step the target structure contains an opening bracket, 
        pair the opposing closing bracket with an opposing nucleotide

        """
        ind1 = self.padded_index
        ind2 = self.current_nucleotide

        self.mat[action, ind1] = 1

        if self.target.seq[ind2] == '(':
            pair = self.target.paired_sites[ind2] + self.kernel_size
            index = self.reverse_action[action] 
            self.mat[index, pair] = 1
    
    def action(self, action):
        """
        Perform an appropriate action based on config
        """
    
    def find_next_unfilled(self):
        """
        Go to the next unfilled nucleotide
        """
        count = 1
        string = self.string[self.current_nucleotide:]
        while True:
            if count >= self.target.len - self.current_nucleotide - 1:
                break
            if string[count] == '-':
                break
            count += 1
        
        self.current_nucleotide += count
        self.padded_index += count

    def hamming_distance(self, seq1, seq2):
        matches = [char1 != char2 for char1, char2 in zip(seq1, seq2)]
        self.hd = sum(matches)
        return self.hd

    def evaluate(self, env_id, verbose=False):
        """
        Evaluate the current solution, measure the hamming distance between the folded structure and the target
        """
        self.folded_design, self.fe = fold_fn(self.string, env_id)
        self.hd = self.hamming_distance(self.target.seq, self.folded_design)
        r = (1 - float(self.hd)/float(self.target.len)) ** self.reward_exp
        if verbose:
            print('\nFolded sequence : \n {} \n Target: \n   {}'.format(self.folded_design, self.target.seq))
            print('\nHamming distance: {}\n'.format(r))
        return r
    
    def get_state(self, reshape=False):
        """
        Return the current state of the solution (window around the current nucleotide)
        """

        if self.use_padding:
            start, end = self.padded_index - self.kernel_size, self.padded_index + self.kernel_size
            state = np.vstack([self.ptarget[:, start:end], self.mat[:, start:end]])

            if reshape:
                state = np.reshape
            else:
                state = np.expand_dims(state, axis=2)


        else:
            start = np.clip(self.current_nucleotide - self.kernel_size, 0, self.target.len - self.kernel_size*2-1)
            target = self.target.bin[start:start+self.kernel_size*2]

            if self.use_full_state:
                current = self.mat[:, start:start+self.kernel_size*2]
                state = np.vstack([target,current])
                state = np.expand_dims(state, axis=2)
            else:
                state = target
                state = np.expand_dims(state, axis=1)
                state = np.expand_dims(state, axis=2)
        return state
    
    def summary(self, print_output=False, colorize=True):
        """
        Formatted summary of the solution
        """
        date = datetime.datetime.now().strftime("%m-%d_%H-%M")
        separator = '\n\n' + ''.join(["*"] * 20) + ' ' + date + ' ' + ''.join(["*"] * 20)
        header = '\n\nSeq Nr. {:5}, Len: {:3}, DBr: {:.2f}, HD: {:3}, Reward: {:.5f}'.format(self.target.file_nr, self.target.len, self.target.db_ratio, self.hd, self.r)

        solution = 'S:  ' + self.string
        folded   = 'F:  ' + self.folded_design
        target   = 'T:  ' + self.target.seq
        free_en  = 'FE: {:.3f}'.format(self.fe)
        summary = [header, solution, folded, target, free_en]
        if colorize:
            colored  = 'S:  ' + colorize_nucleotides(self.string)
            mm1, mm2 = highlight_mismatches(self.folded_design, self.target.seq)
            hl1 = 'F:  ' + mm2
            hl2 = 'T:  ' + mm1
            colored_summary = [header, colored, hl1, hl2, free_en]
        else: colored_summary = summary
        if print_output:
            for line in colored_summary:
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