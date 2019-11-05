import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as c
import numpy as np

# meth = ['RLfold', 'RNAInverse', 'Learna', 'Nupack', 'antaRNA'][::-1]
# methods = 5
# n = 50
# results = np.zeros([methods*3, n])
# simple_results = np.random.random([methods, n])

# for j in range(n):
#     for i in range(methods):
#         ind = i*3
#         results[ind:ind+2, j] = np.random.random()

# times = np.zeros([8, 29])
# repeats = 1
# methods = ['RNAinverse', 'MODENA', 'NUPACK', 'antaRNA', 'rnaMCTS', 'LEARNA', 'rlfold', 'mrlfold']

def get_times(filename):
    data = []
    for filename in filenames:
        with open(filename, 'r') as f:
            data += f.readlines()

    current = -1
    for i, line in enumerate(data):
        if line[0] in ['-', 'A', 'C', 'G', 'U']:
            stats = line.split(',')
            # seq, source, time = stats[:3]
            source = stats[1].strip()
            if source not in methods:
                methods.append(source)
            t = float(stats[2].strip())
            if source == 'LEARNA': t *= 2
            if stats[3] == '':
                t = 65
                hd = 100
                fe = 0
                ed=.5
            else:
                hd = int(stats[3])
                fe = float(stats[5])
                ed = float(stats[6])
                if hd > 0:
                    t = 60
            table[methods.index(source), current//repeats] += np.array([t*2/repeats, hd//repeats, fe/repeats, ed/repeats])
        elif line[0] == 's':
            print(current)
            current += 1

    return table


methods = []
repeats = 2
n_methods = 9
seqs = 102
dim=0
f1 = '/home/andrius/thesis/code/rlfold/results/comparison/eterna_4201/solutions.csv'
f2 = '/home/andrius/thesis/code/rlfold/results/comparison/eterna_2033/solutions.csv'
f3 = '/home/andrius/thesis/code/rlfold/results/comparison/rfam_taneda_6136/solutions.csv'

filenames = [f1]

table = np.zeros([n_methods, seqs, 4])
kwargs = {}

results = get_times(filenames)

data = results[:, :, dim]

# results = np.clip(results, 0, 180)            # print(line, stats)
l = [[(0,1,0),(1, .5, 0),(0, 0, 0)],#(0.2, 0, 0),(0, 0.5, 0),,(0,0.5,0), (.5,.5,0), (0.5,0,0),
     [(0,1,0),(1, .5, 0),(1, 1, 1)],
     [(0,0,0), (1, 1, 1)],
     [(0,1,0),(0, 1, 1),(0, 0, 0)]]

if dim == 3:
    kwargs = dict(norm=c.LogNorm(vmin=data.min()+0.01, vmax=data.max()))
if dim == 0:
    kwargs = dict(norm=c.LogNorm(vmin=0.1, vmax=data.max()))

# def colormatrix(data, labels)

cmap=LinearSegmentedColormap.from_list('b', l[dim], 24)
plt.pcolor(data, edgecolors='k', linewidths=5, cmap=cmap, **kwargs)
plt.yticks(np.arange(0.5, n_methods+0.5, 1), labels=methods)
plt.ylabel('Method')
plt.xticks(np.arange(0.5, seqs+.5, 1), labels=range(1,seqs+1))
plt.xlabel('Sequence #')
cbar = plt.colorbar() #orientation='horizontal'
cbar.set_label('Time taken.', rotation=270)
# plt.legend('Time taken')
plt.show()





# plt.imshow(results, origin='upper', cmap=cmap)
# plt.ylim(0,9)
# plt.yticks(range(1, len(methods)), labels=methods)
# plt.xticks(range(1, 29))
# plt.colorbar(shrink=0.33)
# plt.show()
# plt.cla()