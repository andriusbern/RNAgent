import numpy as np

class Sequence(object):
    """
    Dot-bracket notation sequence of RNA secondary structure
    Contains methods for preprocessing, etc.
    """
    def __init__(self, sequence, file_id=None, file_nr=None, encoding_type=0):
        self.seq = sequence
        self.len = len(sequence)
        self.bin = self.to_binary()
        self.file_id = file_id
        self.file_nr = file_nr
        self.db_ratio = float(sum([1 if x == '.' else 0 for x in self.seq])) / self.len
        self.loops = self._count_loops()
        self.n_loops = len(self.loops)
        self.paired_sites = self.find_complementary()
        self.markers, self.counter = self.sequential_parsing()
        self.encoding_type = encoding_type
        self.structure_encoding = self.to_matrix()
        
    def __repr__(self):
        return self.seq

    def summary(self):
        msg = ''
        msg += 'Seq: {}, Len: {}, DBR: {}, Loops: {}'.format(self.file_id, self.len, self.db_ratio, self.n_loops)
        print(msg)

    def to_binary(self):
        """
        Convert 
        """
        mapping = {'.': 0, '(':1, ')':1}
        binary = [mapping[self.seq[x]] for x in range(len(self.seq))]
        return np.array(binary)

    def to_matrix(self):
        """
        Convert the current sequence into a encoded format:

        ..((..))..  ---.
                       |
        1100110011     |
        0011000000 <---`
        0000001100
        """
        # Dots and brackets only
        if self.encoding_type == 0:
            mapping = {'.': 0, '(':1, ')':1}
            template = self.seq
            rows = 2

        if self.encoding_type == 1:
            mapping = {'.': 0, '(':1, ')':2}
            template = self.seq
            rows = 3

        # Openings, closings, internal loops, hairpin loops, multiloops, ends
        if self.encoding_type == 2:
            mapping = {'O': 0, 'C': 1, 'I': 2, 'H': 3, 'M':4, 'E':5}
            template = self.markers
            rows = 6
        
        # Brackets, internal loops, hairpin loops, multiloops
        if self.encoding_type == 3:
            mapping = {'O': 0, 'C': 0, 'I': 1, 'H': 2, 'M':3, 'E':3}
            template = self.markers
            rows = 4
        
        # Create encoding
        matrix = np.zeros([rows, self.len])
        for index in range(self.len):
            try:
                matrix[mapping[template[index]], index] = 1
            except KeyError:
                pass

        return matrix
             
    def find_complementary(self):
        """
        Find complementary nucleotide indices by expanding outwards
        from hairpin loop edges
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

    #########
    # Metrics

    def _count_loops(self):
        """
        Find the loop indices in the sequence
        """
        loops = []
        opening = False
        for index in range(self.len-1):
            if self.seq[index] == '(':
                opening = True
                last_opening = index
            if opening:
                if self.seq[index] == ')':
                    loops.append([last_opening, index])
                    opening = False
        return loops  

    def sequential_parsing(self):
        """
        Creates markers for 
            1. Openings and closings [ (   ) ]
            1. Multiloops, hairpin loops, internal loops/mismatches, dangling ends
        """
        markers = ['N'] * self.len
        counter = dict(M=0, H=0, I=0, E=0)
        prev_bracket = '('
        consecutive = 0

        for index in range(self.len):
            if self.seq[index] == '.':
                consecutive += 1
            else:
                if self.seq[index] == '(':
                    markers[index] = 'O' # Opening
                if self.seq[index] == ')':
                    markers[index] = 'C' # Closing

                if consecutive > 0:
                    # Hairpin loop
                    if self.seq[index] == ')' and prev_bracket == '(':
                        markers[index-consecutive:index] = 'H' * consecutive
                        counter['H'] += 1

                    # Internal loop / mismatches
                    if self.seq[index] == ')' and prev_bracket == ')':
                        markers[index-consecutive:index] = 'I' * consecutive
                        counter['I'] += 1

                    # Internal loop / mismatches
                    if self.seq[index] == '(' and prev_bracket == '(':
                        markers[index-consecutive:index] = 'I' * consecutive
                        counter['I'] += 1
                    
                    # Multiloop
                    if self.seq[index] == '(' and prev_bracket == ')':
                        markers[index-consecutive:index] = 'M' * consecutive
                        counter['M'] += 1

                consecutive = 0
                prev_bracket = self.seq[index]

        # Dangling ends
        for index in range(self.len):
            if self.seq[index] == '(':
                break
            else:
                markers[index] = 'E'
                counter['E'] += 1
        for index in reversed(range(self.len)):
            if self.seq[index] == ')':
                break
            else:
                markers[index] = 'E'
                counter['E'] += 1
            
        return ''.join(markers), counter