import numpy as np
import argparse, os
from rlif.settings import ConfigManager as settings
from glob import glob


def get_times(filename):
    data = []
    for filename in filenames:
        with open(filename, 'r') as f:
            data += f.readlines()

    # Count number of different methods
    methods = []
    i = 1
    while True:
        i += 1
        line = data[i]
        if len(line.split(',')[0]) < 2:
            break
        methods.append(line.split(',')[1])
    n_methods = len(methods)    

    # Count the number of times same target seqs are used
    current = -1
    l = data[0].split(',')[0]
    repeats = 1
    seqs = 0
    tlim =  int(data[0].split(',')[1])
    for i, line in enumerate(data[1:50]):
        if l == line.split(',')[0]:
            repeats +=1

    # Count the number of different sequences in the log
    for i, line in enumerate(data[1:]):
        if line[0] == 's':
            seqs += 1./repeats

    seqs = round(seqs)
    print(methods, seqs, repeats)
    table = np.zeros([n_methods, seqs, 4])

    # Parse the log
    for i, line in enumerate(data):
        if line[0] in ['-', 'A', 'C', 'G', 'U']:
            stats = line.split(',')
            source = stats[1].strip()
            if source not in methods:
                methods.append(source)
            t = float(stats[2].strip())
            if stats[3] == '':
                t = tlim
                hd = 100, 0, .5
            else:
                hd, fe, ed = int(stats[3]), float(stats[5]), float(stats[6])
                if hd > 0:
                    t = tlim
            table[methods.index(source), current//repeats] += np.array([t/repeats, hd//repeats, fe/repeats, ed/repeats])
        elif line[0] == 's':
            current += 1

    return table, methods

def cumtime(times):
    """
    Counts the number of sequences solved before some period of time
    """
    ticks = 50
    interval = np.linspace(0.1,np.max(times), ticks)
    methods, seqs = np.shape(times)
    cumulative = np.zeros([methods, ticks])
    for method in range(methods):
        for seq in range(seqs):
            cumulative[method] += times[method, seq] < interval
    
    return cumulative, np.round(interval)
