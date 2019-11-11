import numpy as np
import forgi, math
import matplotlib.pyplot as plt
from rlif.utils import create_browser, show_rna

class DotBracket(object):
    """
    Container for dot-bracket annotated RNA secondary structure sequences
    Contains methods for preprocessing, etc.
    """
    def __init__(self, sequence, file_id=None, file_nr=None, encoding_type=2, graph_based=False):
        self.encoding_type = encoding_type
        self.file_nr = file_nr
        self.file_id = file_id
        self.seq = sequence
        self.len = len(sequence)

        # Parse
        self.loops = self._count_loops()
        self.base_pair_indices = self.find_base_pairs()
        self.rev_base_pair_indices = {v: k for k, v in self.base_pair_indices.items()}
        self.struct_motifs, self.counter = self.parse_structure()
        self.structure_encoding = self.to_matrix()
        self.percent_unpaired = float(sum([1 if x == '.' else 0 for x in self.seq])) / self.len
        
    def __repr__(self):
        return self.seq

    def summary(self):
        msg = 'Seq: {:5}, Len: {:4}, DBR: {:.2f}, Loops: {:3}'.format(self.file_id, self.len, self.percent_unpaired, len(self.loops))
        return(msg)

    def to_binary(self):
        """
        Convert dot-brackets into binary format numpy array
        """
        mapping = {'.': 0, '(':1, ')':1}
        return np.array([mapping[self.seq[x]] for x in range(len(self.seq))])

    def to_matrix(self):
        """
        Convert the current sequence into a encoded format:
        e.g.
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

        # Openings, closings, internal loops, next_hairpin loops, multiloops, ends
        if self.encoding_type == 2:
            mapping = {'O': 0, 'C': 1, 'I': 2, 'H': 3, 'M':4, 'E':5}
            template = self.struct_motifs
            rows = 6
        
        # Brackets, internal loops, next_hairpin loops, multiloops
        if self.encoding_type == 3:
            mapping = {'O': 0, 'C': 0, 'I': 1, 'H': 2, 'M':3, 'E':3}
            template = self.struct_motifs
            rows = 4
        
        # Multidiscrete encoding
        if self.encoding_type == 4:
            mapping = {'O': 1, 'C': 2, 'I': 3, 'H': 4, 'M':5, 'E':6}
            encoding = np.array([mapping[x] for x in self.struct_motifs])
            return encoding

        # Two-strand encoding
        if self.encoding_type == 5:
            mapping = {'O': 1, 'C': 2, 'I': 3, 'H': 4, 'M':5, 'E':6}
            encoding = np.array([[mapping[self.struct_motifs[x-1]] if x!=0 else 0 for x in self.strand1], 
                                 [mapping[self.struct_motifs[x-1]] if x!=0 else 0 for x in self.strand2]])
            return encoding

        # Create a one-hot encoding
        encoding = np.zeros([rows, self.len])
        for index in range(self.len):
            element = template[index]
            encoding[mapping[element], index] = 1

        return encoding

    def find_base_pairs(self):
        """
        Find paired nucleotide indices by expanding outwards
        from next_hairpin loop edges
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
            max_count = 0
            while max_count < 1000:
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
                max_count += 1
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

    def parse_structure(self):
        """
        Creates markers for 
            1. Openings and closings
            1. Multiloops, hairpin loops, internal loops/mismatches, dangling ends
        """
        markers = ['N'] * self.len
        counter = dict(M=0, H=0, I=0, E=0)
        prev_bracket = 'X'
        motif_length = 0
        mapping = {'()':'H', ')(':'M', '((':'I', '))':'I', 'X(':'E', ')X':'E', 'XX':'E'}

        def mark(motif, index, length):
            """
            """
            markers[index-length:index] = motif * length
            counter[motif] += 1

        dotbracket = self.seq + 'X' # DotBracket with X at the end
        for i, symbol in enumerate(dotbracket):
            if symbol == '.':
                motif_length += 1
            elif symbol == 'X':
                motif = mapping[prev_bracket+symbol]
                mark(motif, i, motif_length)
            else:
                if symbol == '(':
                    markers[i] = 'O' # Opening
                if symbol == ')':
                    markers[i] = 'C' # Closing

                # Mark previous structural motif
                if motif_length > 0:
                    motif = mapping[prev_bracket+symbol] # Motif identifier
                    mark(motif, i, motif_length)
                
                motif_length = 0
                prev_bracket = symbol
            
        return ''.join(markers), counter
    
    def get_graph(self):
        """
        Parse the graph using forgi
        Store the element representations
        """
        
        strand_dict = {}
        graph, = forgi.load_rna(self.seq)
        for element, indices in graph.defines.items():
            dimensions = list(graph.get_node_dimensions(element))
            if dimensions[1] == 1000 or dimensions[1] == -1:
                dimensions[1] = 0
            length = max(dimensions)
            strand1 = [0] * length
            strand2 = [0] * length
            description = [element[0]] * length
            # Stem ends
            if element[0] == 's':
                description[0], description[-1] = 'S', 'S'

            shift1, shift2 = 0, 0
            if len(indices) == 2:
                count1 = indices[1] - indices[0]
                if dimensions[0] > dimensions[1]:
                    for i in range(length):
                        strand1[i] = indices[0] + i
                else:
                    for i in range(length):
                        strand2[i] = indices[1] - i
            if len(indices) == 4:
                count1 = indices[1] - indices[0]
                count2 = indices[3] - indices[2]

                if count1 > count2: shift2 = (count1 - count2) // 2
                else: shift1 = (count2 - count1) // 2

                for i in range(dimensions[0]):
                    strand1[shift1 + i] = indices[0] + i
                for i in range(dimensions[1]):
                    strand2[shift2 + i] = indices[3] - i

            strand_dict[element] = [strand1, strand2, description]

        return strand_dict, graph

    def create_strands(self, dic, graph):
        """
        
        """
        current_element = graph.get_elem(1)
        last_el  = graph.get_elem(self.len)
        failed = False
        n_hairpins = len(list(graph.hloop_iterator()))
        visited = {}
        for key in dic.keys():
            visited[key] = 0

        first = dic[current_element][0]
        second = dic[current_element][1]
        description = dic[current_element][2]
        visited[current_element] += 1

        try:
            if n_hairpins > 0:
                elements = list(graph.hloop_iterator())
                elements.append(last_el)
                for next_hairpin in elements:
                    path = graph.shortest_path(current_element, next_hairpin)
                    for element in path[1:]:

                        if visited[element] == 0:
                            first = first + dic[element][0]
                            second = second + dic[element][1]
                        else:
                            first = first + dic[element][1][::-1]
                            second = second + dic[element][0][::-1]

                        visited[element] += 1
                    current_element = next_hairpin

        except Exception as e:
            print(e)
            failed = True

        return first, second, failed

    
    def get_subgraph(self, nucleotide):
        """
        Gets the subgraph surrounding the current nucleotide

        Returns a networkx subgraph
        """

        element = self.graph.get_elem(nucleotide)
        connected = self.graph.connections(element)

        # In case of a multiloop or external loop add components surrounding the flanking stems as well
        if element[0] == 'm' or element[0] == 't' or element == 'f':
            try:
                connected += self.graph.connections(connected[0]) + self.graph.connections(connected[1])
            except:
                pass
        connected += [element]

        nucleotides = self.graph.elements_to_nucleotides(connected)
        nx_graph = self.graph.to_networkx()
        subgraph = nx_graph.subgraph(nucleotides)

        return subgraph

    def visualize(self):
        driver = create_browser('dataset')
        show_rna(self.seq, None, driver=driver, html='dataset')
        print(self.summary())
