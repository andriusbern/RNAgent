import numpy as np
import rlfold.settings as settings
import os, random
import matplotlib.pyplot as plt
import copy, datetime, yaml
from .sequence import Sequence


def hamming_distance(seq1, seq2):
    """
    Hamming distance between two strings
    Returns hamming distance and mismatch indices
    """
    matches = [char1 != char2 for char1, char2 in zip(seq1, seq2)]
    hd = sum(matches)
    indices = np.where(matches)[0]
    return hd, indices

def colorize_nucleotides(sequence):
    """
    Color coding for the console output for nucleotide sequences
    """
    mapping = {'A':'\033[2;30;42mA\033[0m',
               'U':'\033[2;30;46mU\033[0m',
               'G':'\033[2;30;41mG\033[0m',
               'C':'\033[2;30;45mC\033[0m',
               '-':'\033[2;37;40m-\033[0m'}    
    return ''.join([mapping[x] for x in sequence])

def highlight_mismatches(seq1, seq2):
    """
    Color highlighting of mismatches between strings
    """
    mseq1, mseq2 = '', '' # Modified sequences
    for char1, char2 in zip(seq1, seq2):
        if char1 != char2: # Mismatch
            mseq1 += '\033[5;36;41m{}\033[0m'.format(char1)
            mseq2 += '\033[5;36;41m{}\033[0m'.format(char2)
        else:              # Match
            mseq1 += '\033[2;32;40m{}\033[0m'.format(char1)
            mseq2 += '\033[2;32;40m{}\033[0m'.format(char2)

    return mseq1, mseq2

def load_sequence(num, dataset='rfam_learn_train', encoding_type=0):
    """
    Load a sequence.rna file and return a Sequence object
    """
    print('Loading sequence #{}...'.format(num), end='\r')
    path = os.path.join(settings.DATA, dataset)
    filename = os.path.join(path, '{}.rna'.format(num))
    try:
        with open(filename, 'r') as f:
            seq = f.readline()[:-1]
        return Sequence(seq, filename, num, encoding_type=encoding_type)
    except FileExistsError:
        print('File {} does not exist...'.format(filename))
        return []
    
def load_length_metadata(dataset, length):
    """
    Loads length metadata
    """
    filename = os.path.join(settings.DATA, 'metadata', dataset, 'len', '{}.yml'.format(length))
    try:
        with open(filename, 'r') as f:
            data = yaml.load(f)
        return data
    except:
        return []
