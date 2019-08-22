import numpy as np
import forgi, math
import matplotlib.pyplot as plt
from rlfold.interface import create_browser, show_rna

class Sequence(object):
    """
    Dot-bracket notation sequence of RNA secondary structure
    Contains methods for preprocessing, etc.
    """
    def __init__(self, sequence, file_id=None, file_nr=None, encoding_type=0, graph_based=False):
        self.seq = sequence
        self.len = len(sequence)
        self.bin = self.to_binary()
        self.db_ratio = float(sum([1 if x == '.' else 0 for x in self.seq])) / self.len
        self.loops = self._count_loops()
        self.n_loops = len(self.loops)
        self.paired_sites = self.find_complementary()
        self.markers, self.counter = self.sequential_parsing()
        self.encoding_type = encoding_type
        self.structure_encoding = self.to_matrix()
        self.file_id = file_id
        self.file_nr = file_nr

        # if graph_based:
        self.graph = forgi.graph.bulge_graph.BulgeGraph.from_dotbracket(self.seq, dissolve_length_one_stems=False)
            # self.element_dict = self.get_graph()
            # self.primary, self.secondary, self.graph_markers = self.create_strands(self.element_dict)
            # self.graph_based_encoding = self.get_graph_based_encoding()
        
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

        # Openings, closings, internal loops, next_hairpin loops, multiloops, ends
        if self.encoding_type == 2:
            mapping = {'O': 0, 'C': 1, 'I': 2, 'H': 3, 'M':4, 'E':5}
            template = self.markers
            rows = 6
        
        # Brackets, internal loops, next_hairpin loops, multiloops
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

    def get_graph_based_encoding(self):
        """

        """
        if self.encoding_type == 2:
            mapping = {'S': -2, 's': -1, 'h':1, 'i':2, 'f':3, 'm':3, 't':3}

        matrix = np.zeros([3, len(self.graph_markers)])
        for index in range(self.len):
            matrix[0,index] = mapping[self.graph_markers[index]]

        matrix[1,:] = self.primary
        matrix[2,:] = self.secondary

        return matrix
             
    def find_complementary(self):
        """
        Find complementary nucleotide indices by expanding outwards
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

    def sequential_parsing(self):
        """
        Creates markers for 
            1. Openings and closings [ (   ) ]
            1. Multiloops, next_hairpin loops, internal loops/mismatches, dangling ends
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
    
    def get_graph(self):
        """
        Parse the graph using forgi
        Store the element representations
        """
        
        # elements = self.graph.defines
        # start = elements.get('f0')
        # end   = elements.get('t0')

        # if start is not None and end is not None:

        # Implement start and end joining if necessary

        strand_dict = {}
        for element, indices in self.graph.defines.items():
            dimensions = list(self.graph.get_node_dimensions(element))
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
            # try:
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


            # except Exception as e: 
            #     print(e)

            strand_dict[element] = [strand1, strand2, description]

        return strand_dict

    def create_strands(self, dic):
        """
        
        """
        current_element = self.graph.get_elem(1)
        last_el  = self.graph.get_elem(self.len)
        n_hairpins = len(list(self.graph.hloop_iterator()))
        visited = {}
        for key in dic.keys():
            visited[key] = 0

        first = dic[current_element][0]
        second = dic[current_element][1]
        description = dic[current_element][2]
        visited[current_element] += 1

        try:
            if n_hairpins > 0:
                elements = list(self.graph.hloop_iterator())
                elements.append(last_el)
                for next_hairpin in elements:
                    
                    path = self.graph.shortest_path(current_element, next_hairpin)
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

        return first, second, description

    
    def get_subgraph(self, nucleotide):
        """
        Gets the subgraph surrounding the current nucleotide

        Returns a networkx subgraph
        """

        element = self.graph.get_elem(nucleotide)
        connected = self.graph.connections(element)

        # In case of a multiloop or external loop add components surrounding the flanking
        # stems as well
        if element[0] == 'm' or element[0] == 't' or element == 'f':
            try:
                connected += self.graph.connections(connected[0]) + self.graph.connections(connected[1])
            except:
                pass
        connected += [element]

        # for element in connected:
        #     if element[0] == 'm' and self.graph.get_length(element) == 0:
        #         connected += self.graph.connections(element)

        nucleotides = self.graph.elements_to_nucleotides(connected)
        nx_graph = self.graph.to_networkx()
        subgraph = nx_graph.subgraph(nucleotides)

        return subgraph

    def visualize(self):
        driver = create_browser('dataset')
        show_rna(self.seq, None, driver=driver, html='dataset')
        print(self.summary())





        



