import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as c
import numpy as np
# from rlif.utils.plots import get_times, cumtime
import numpy as np
import argparse, os
# from rlif.settings import ConfigManager as settings
from glob import glob


def get_times(filename):
    data = []
    methods = []
    for filenam in filename:
        with open(filenam, 'r') as f:
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
    tlim =  int(data[0].split(',')[1]) * tmult
    for i, line in enumerate(data[1:50]):
        if l == line.split(',')[0]:
            repeats +=1

    # Count the number of different sequences in the log
    for i, line in enumerate(data[1:]):
        if line[0] == 's':
            seqs += 1./repeats

    seqs = round(seqs)
    table = np.zeros([n_methods, seqs, 5])
    p=0
    # Parse the log
    for i, line in enumerate(data):
        if line[0] in ['-', 'A', 'C', 'G', 'U']:
            m += 1
            stats = line.split(',')
            source = stats[1].strip()
            
            t = float(stats[2].strip())
            # print(stats)
            try:
                if stats[3] == '':
                    t = tlim
                    hd, fe, ed = 100, 0, .5
            except:
                t = tlim
                hd, fe, ed = 100, 0, .5
            try:
                if float(stats[7]) == 1:
                    p+= 1
                plus = 1 if float(stats[7]) == 1 else 0
                hd, fe, ed = int(stats[3]), float(stats[5]), plus
                if hd > 0:
                    t = tlim
            except:
                t = tlim
                hd, fe, ed = 100, 0, .5
            # ind = methods.index(source) -1 
            # print(ind)
            c = current//repeats
            table[m, c, :] += np.array([t*2/repeats, hd//repeats, fe/repeats, ed/repeats, l])
        elif line[0] == 's':
            m=-1
            l = len(data[i-1].split(',')[0])
            current += 1
    # print(table[:,:,4])
    print(p)
    print(np.sum(table[0,:,3]))
    return table, methods, tlim

def cumtime(times):
    """
    Counts the number of sequences solved before some period of time
    """
    ticks = 121
    interval = np.linspace(0.1,np.max(times), ticks)
    methods, seqs = np.shape(times)
    cumulative = np.zeros([methods, ticks])
    for method in range(methods):
        for seq in range(seqs):
            cumulative[method] += times[method, seq] < interval
    
    return cumulative, np.round(interval)

def colored_table(data, t=False):
    
    # datas = [data.T, data2.T]
    
    n_methods, seqs = data.shape
    l = [[(0,1,.5),(0,1,.5), (0,.8,0),(.25,.5,.25),(.35, .5, .35),(1, 1, 1)],#(0.2, 0, 0),(0, 0.5, 0),,(0,0.5,0), (.5,.5,0), (0.5,0,0),
        [(0,1,0),(1, .5, 0),(1, 1, 1)],
        [(0,0,0), (1, 1, 1)],
        [(0,1,0),(0, 1, 1),(1, 1, 1)]]
    
    if args.dim == 3:
        ticks = [.001, .01, .1 , .5, 1]
        kwargs = dict(norm=c.LogNorm(vmin=data.min()+0.01, vmax=data.max()))
        label = 'Ensemble defect'
    if args.dim == 0:
        ticks = [.1, 1, 10, 60, 120, tlim]
        label = 'Average time taken (s)'
        kwargs = dict(norm=c.LogNorm(vmin=data.min()+0.01, vmax=tlim))

    cmap=LinearSegmentedColormap.from_list('b', l[args.dim], 256)
    # f, axes = plt.subplots(1, 2)
    # plt.show()
    # for i in range(2):
    #     axes[i].pcolor(np.flip(datas[i]), edgecolors='k', linewidths=5, cmap=cmap, **kwargs)
    #     plt.show()
    if t:
        fig = plt.figure(figsize=(20, 4))
        data = data
        
    else:
        fig= plt.figure(figsize=(3, 12))
        data = np.flip(data.T)
    ax = fig.gca()
    kw = dict(linestyle='-',snap=True, capstyle='round', antialiased=True, rasterized=True)
    plt.pcolor(data, edgecolors='w', linewidths=5, cmap=cmap, **kwargs, **kw)
    if t:
        plt.yticks(np.arange(0.5, n_methods+0.5, 1), labels=(methods))
        # plt.ylabel('Method')
        # plt.xticks(np.arange(0.5, seqs+.5, 1), labels=range(1,seqs+1, 1))
        plt.xticks(np.arange(0.5, seqs+.5, 1), labels=range(51,seqs+51, 1))
        plt.xlabel('Sequence #')
        plt.rc('xtick', labelsize=12)
        ax.tick_params(axis='y', labelsize=18)
    else:
        plt.xticks(np.arange(0.5, n_methods+0.5, 1), labels=reversed(methods), rotation=90)
        # plt.xlabel('Method')
        plt.yticks(np.arange(0.5, seqs+.5, 1), labels=reversed(range(1,seqs+1)))
        plt.ylabel('Sequence #')
        # plt.rc('ytick', labelsize=8)    # fontsize of the tick labels
        ax.tick_params(axis='x', labelsize=15)
    cbar = plt.colorbar() #orientation='horizontal'
    cbar.set_label(label, rotation=90)
    # cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticks)
    # plt.legend('Time taken')
    plt.show()



if __name__ == "__main__":
    import argparse, os
    # from rlif.settings import ConfigManager as settings
    from glob import glob

    dim = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--num', type=int)
    parser.add_argument('-p','--plot', type=int, default=0)
    parser.add_argument('-t', '--transpose', action='store_true')
    parser.add_argument('-q', '--title', type=str, default='')
    parser.add_argument('-d', '--dim', type=int, default=0)
    args = parser.parse_args()

    dire = os.path.join('/home/andrius/thesis/rlif/rlif/results', 'comparison')
    subdirs = glob(dire+'/*/')
    for i, sub in enumerate(subdirs):
        print(i, sub, len(glob(sub+'*')))
    filenames = [subdirs[args.num]+'solutions.csv']
    # file1 = [subdirs[15]+'solutions.csv']
    # file2 = [subdirs[17]+'solutions.csv']
    # file1 = [subdirs[3]+'solutions.csv']
    # file2 = [subdirs[6]+'solutions.csv']
    file2 = [subdirs[1]+'solutions.csv']
    file1 = [subdirs[19]+'solutions.csv']
    file3 = [subdirs[40]+'solutions.csv']

    print(filenames)
    # Concat runge

    ## LENGTHS
    tmult = 1
    results, methods, tlim = get_times(filenames)

    # print(buckets, sol, tot, ts)
    # results1, methods, tli = get_times(file1)
    # results2, meth, tlim = get_times(file2)
    # tmult = 1
    # results3, _, _ = get_times(file3)

    # print(tli, tlim)
    # methods = meth + methods

    # # rlif = np.min(results2[:6, :,0], axis = 0)
    # print(results1.shape, results2.shape)

    # mod = np.ones([2, 50]) * tlim
    # mod[:,:45] = results1[:,1:,0]
    # dat = np.vstack([results2[:,:,0],mod])
    # dat = np.hstack([results3[:,:,0], dat])
    # print(file1, file2)
    # results1[:4, :, :] = results1[:4, :, :] / 2
    # results1[0:4, :, 0] = results1[:4, :, 0] / 2 + results2[:4, :, 0]/2 
    # results1[6, :, 0] = rlif
    # results = dat[:, 50:]
    # data = dat[:, 49:]
    # seqs = 50
    data = results[:, :, args.dim]
    data = np.clip(data, 0.089, 10000)
    
    solved = results[:,:] < tlim
    solved = np.sum(solved, axis=1)
    print(solved)
    # results2, methods2 = get_times(filenames2)
    # data2 = results2[:, :, dim]
    try:
        methods[methods.index('rlifold')] = 'RLIF'
        methods[methods.index('rlif')] = 'RLIF'
    except:
        pass

    colors = ['b', 'g', 'r', 'k', 'y', 'dimgray', 'orange', 'p']

    # # Join
    # j = 6
    # data[-j, :] = np.min(data[-j:, :], axis=0)
    # data = data[:-j-1,:]
    # methods = methods[:-j-1]
    methods[-1] = 'RLIF'
    
    # plt.rcParams.update({'font.size': 14})
    if args.plot == 1:
        ts, intervals = cumtime(data)
        for i in range(results.shape[0]):
            plt.plot(ts[i, :])
        plt.xticks(ticks=range(len(intervals))[::10],labels=[int(x) for x in intervals[::10]])
        plt.legend(methods,ncol=4, loc='center', bbox_to_anchor=(0.5, -0.2))
        # plt.legend(methods)
        plt.xlim([0, 59])
        plt.ylim([0, data.shape[1]])
        plt.ylabel('# Sequences solved')
        plt.xlabel('Time (s)')
        plt.yticks(ticks=range(0,data.shape[1]+3,10), labels=range(0,data.shape[1]+3,10))
        plt.grid()
        plt.show()
    elif args.plot ==0:
        colored_table(data, args.transpose)
        # def colormatrix(data, labels)
    elif args.plot == 2:
        cols = [(x/35, x/29 ,x/35) for x in solved]
        tcks = range(1, len(methods)+1)
        plt.bar(x=tcks, height=solved, color=colors[:len(methods)], alpha=0.3)
        plt.xticks(tcks, labels=methods)
        plt.yticks(ticks=range(0,data.shape[1]+3,5), labels=range(0,data.shape[1]+3,5))
        # plt.xlabel('Method')
        plt.ylim([0,101])
        plt.grid()
        plt.ylabel('# Sequences solved')
        plt.title(args.title)
        plt.show()

    elif args.plot == 3:
        t = results[:,:,0]
        l = results[:,:,4]

        sorts = np.argsort(l[0, :])
        # print(sorts)


        for i in range(2):#range(len(methods)):
            plt.plot(l[i, sorts], t[i, :])

        plt.show()
    elif args.plot == 4:
        buckets = np.arange(50, 475, 25)
        sh = buckets.shape
        sol = np.zeros(sh)
        tot = np.zeros(sh)
        ts = np.zeros(sh)
        tlist = [[] for _ in range(sh[0])]
        tlim = 30

        for i in range(results.shape[1]):
            for j in range(1, len(buckets)):
                l = results[:,i,4]
                if buckets[j-1]<l <buckets[j]:
                    # print(l)
                    t = results[:,i,0]
                    if t < tlim:
                        sol[j-1] += 1
                    tot[j-1] += 1
                    ts[j-1] += t
                    tlist[j-1].append(t)
        print(sol, tot)
        print(sol.sum())
        print(tot.sum())
        perc = sol[:-1]/tot[:-1] * 100
        # perc = ts[:-1]/tot[:-1]
        # plt.plot(buckets[:-1], perc)
        stds = [np.std(l) for l in tlist]
        # plt.errorbar(buckets[:-1]+12, y=perc,  alpha=0.5, yerr = stds[:-1])
        plt.bar(buckets[:-1]+12, height=perc, width=20, alpha=0.2, color='green')
        plt.plot(buckets[:-1]+12, perc, color='green')
        plt.ylim([0, 100]) 
        plt.xlabel('Sequence length.')
        plt.ylabel('Percentage of sequences solved %')
        yticks = np.arange(0,110,10)
        plt.yticks(ticks=yticks,labels=yticks)
        # plt.xticks(ticks=buckets, labels=buckets)
        plt.grid()
        plt.show()
        # plt.hist(results)





# plt.imshow(results, origin='upper', cmap=cmap)
# plt.ylim(0,9)
# plt.yticks(range(1, len(methods)), labels=methods)
# plt.xticks(range(1, 29))
# plt.colorbar(shrink=0.33)
# plt.show()
# plt.cla()