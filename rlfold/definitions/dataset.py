import numpy as np
import os, random, yaml, time
import matplotlib.pyplot as plt
from rlfold.definitions import load_length_metadata, load_sequence
from rlfold.definitions import Sequence
import rlfold.settings as settings
from rlfold.interface import show_rna, create_browser

class Dataset(object):
    def __init__(
            self, 
            length=None,
            start=1,
            n_seqs=25000,
            sequences=None,
            dataset='rfam_learn_train',
            encoding_type=0
        ):
        
        self.dataset = dataset
        self.path = os.path.join(settings.DATA, dataset)
        self.sequence_length = length

    
        # Load sequences of specific length only
        if length is not None:
            if isinstance(length, int):
                sequence_file_numbers = load_length_metadata(dataset, length)
                self.sequences = [load_sequence(index, dataset=dataset, encoding_type=encoding_type) for index in sequence_file_numbers[:n_seqs]]
            if isinstance(length, list):
                if len(length) <= 2:
                    length = range(length[0], length[1])
                self.sequences = []
                for l in length:
                    sequence_file_numbers = load_length_metadata(dataset, l)    
                    self.sequences += [load_sequence(index, dataset=dataset, encoding_type=encoding_type) for index in sequence_file_numbers[:n_seqs]]
        
        # Create dataset from a list of sequences loaded elsewhere
        if sequences is not None:
            self.sequences = sequences
        # Otherwise load all consecutive sequences from start to start+n_seqs
        elif sequences is None and length is None:
            self.sequences = [load_sequence(index, dataset=dataset, encoding_type=encoding_type) for index in range(start, start+n_seqs)]
            self.sequences = [index for index in self.sequences if index is not None]

        self.n_seqs = len(self.sequences)
        self.statistics()

    def __repr__(self):
        return self.statistics()

    def __getitem__(self, index):
        return self.sequences[index]

    def visualize(self, auto=False):
        """
        Call the forna container and visualize the dataset
        """
        driver = create_browser('dataset')
        for seq in self.sequences:
            show_rna(seq.seq, None, driver=driver, html='dataset')
            print(seq.summary())
            if not auto: input()
            else: time.sleep(2)

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

    def length_grouping(self):
        """
        Write metadata about sequence lengths for faster loading in the future
        """
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

    def create_subgraph_dataset(self, name):
        """
        Creates a new dataset out of all combinations of subraphs existing
        in the dataset
        """
        directory = os.path.join(settings.DATA, name)
        os.makedirs(directory)

        for sequence in self.sequences:
            element = sequence.graph.get_elem(1)
            subgraph = sequence.get_subgraph(1)
            for i in range(1, len(sequence)):
                current_element = sequence.graph.get_elem(i)

                if current_element != element:
                    current_subgraph = sequence.get_subgraph(i)
                    element = current_element

                sequence.get_subgraph()




