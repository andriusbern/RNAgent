import numpy as np
from rlif.settings import ConfigManager as settings
import os, random
import copy, datetime, yaml
import rlif
import sys, subprocess, time
from .dotbracket import DotBracket


def load_fasta(filename, config=None):
    """
    Returns a list of Solution Objects obtained by reading a fasta file
    """
    from .solution import Solution
    fname = filename.split(settings.delimiter)[-1]
    with open(filename, 'r') as fasta:
        seq = ''
        sequences = []
        for line in fasta.readlines():
            if line[0] == '>':
                name = line[1:].strip('\n')
                if seq != '':
                    seq = seq.replace('T', 'U')
                    structure, fe = settings.fold_fn(seq)
                    target = DotBracket(structure)
                    target.name = name
                    target.nucleotides = seq
                    solution = Solution(target=target, config=config, string=seq, time=0, source=fname)
                    sequences.append(solution)
                    seq = ''
            else:
                seq += line.strip('\n')

    return sequences


def write_fasta(filename, sequences, write_dot_brackets=False, linebreak=80):
    """
    Write a fasta file from a list of Solution Objects
    >Name
    AUGUACGA
    ...

    Alternatively write a Vienna format file with the target dot-bracket structure
    alongside the nucleotide sequence:
    >Name
    AUGUACGA
    ((....))
    ...
    """
    with open(filename, 'a+') as fasta:
        if fasta.readlines().count() > 1:
            fasta.write('\n')

        for solution in sequences:
            name = '>' + sequence.target.name + '\n'
            linebreaks = np.arange([0, len(solution.string), linebreak])
            nucleotide_string = [solution.string[i:i+linebreak]+'\n' for i in linebreaks]
            fasta.write(name)
            fasta.write(nucleotide_string)
            if write_dot_brackets:
                dotbrackets = [solution.target.seq[i:i+linebreak]+'\n' for i in linebreaks]
                fasta.write(dotbrackets)
            fasta.write('\n')



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

def colorize_motifs(sequence):
    """
    Color coding for the console output for nucleotide sequences
    """
    mapping = {'M':'\033[2;30;42mA\033[0m',
               'H':'\033[2;30;46mU\033[0m',
               'I':'\033[2;30;41mG\033[0m',
               'O':'\033[2;30;45mC\033[0m',
               'C':'\033[2;37;43m-\033[0m',
               'E':'\033[2;37;47m-\033[0m'}
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

def load_sequence(num, dataset='rfam_learn', encoding_type=0):
    """
    Load a sequence.rna file and return a DotBracket object
    """
    path = os.path.join(settings.DATA, dataset)
    filename = os.path.join(path, '{}.rna'.format(num))
    try:
        with open(filename, 'r') as f:
            seq = f.readline()[:-1]
            target = DotBracket(seq, filename, num, encoding_type=encoding_type)
        print('Loading sequence #{}...'.format(target.summary()), end='\r')
        return target
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







