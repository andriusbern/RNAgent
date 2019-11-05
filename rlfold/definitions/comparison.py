import numpy as np
import rlfold.settings as settings
import os, random
import copy, datetime, yaml
import rlfold
import sys, subprocess, time
from rlfold.definitions import set_vienna_params
set_vienna_params(2)

sys.path.append('/home/andrius/thesis/code/comparison/learna/src')
sys.path.append('/home/andrius/thesis/code/comparison')
from mcts import fold as mf
from learna import Learna_fold
from rlfold.definitions import Sequence, Solution, Dataset
from rlfold.baselines import SBWrapper

lfold = Learna_fold()
wrapper  = SBWrapper('RnaDesign', 'experiment4').load_model(2, checkpoint='8', n_envs=1)
mwrapper = SBWrapper('RnaDesign', 't238').load_model(1, checkpoint='10', n_envs=1)
bwrapper = SBWrapper('RnaDesign', 'experiment4').load_model(2, checkpoint='8', n_envs=6)
bwrapper.env.set_attr('boosting', True)



def nupack_fold(sequence, file_n=0, time_limit=10):

    path = '/home/andrius/thesis/comparison/nupack3.0.6/bin/'
    fs = str(file_n) + '.fold'
    filename = os.path.join(path, fs)
    with open(filename, 'w') as f:
        f.write(sequence)
    command = ['./design', str(file_n)]
    p = subprocess.Popen(command, cwd=path, stdout=subprocess.PIPE)

    t0, t = time.time(), 0
    while p.poll() is None and t < time_limit:
        t = time.time() - t0

    if p.returncode is not None:
        output = p.communicate()[0].decode('utf-8')
        seq = output.split('\n')[-3].strip()
        seq = seq.split(':')[-1].strip()
        return seq
        
    else:
        p.terminate()
        p.kill()
        return None

def modena_fold(sequence, file_n=0, time_limit=10):

    msg = \
"""{}

;
-1*((F:CONT-50)^2)^0.5
-1*(C:FE-B:EFE)
;
B RNAfold-p 1 "-d2"
C RNAeval 1 "-d2"
F GC 0""".format(sequence)

    path = '/home/andrius/thesis/comparison/MODENA'
    fs = str(file_n) + '.inp'
    filename = os.path.join(path, fs)
    with open(filename, 'w') as f:
        f.write(msg)
    command = ['./modena', '-f', fs]
    p = subprocess.Popen(command, cwd=path, stdout=subprocess.PIPE)
    t0, t = time.time(), 0
    while p.poll() is None and t < time_limit:
        t = time.time() - t0

    if p.returncode is not None:
        output = p.communicate()[0].decode('utf-8')
        res = output.split('\n')

        return res[-7]
    else:
        p.terminate()
        p.kill()
        return None

def antarna_fold(sequence, time_limit=10):

    path = '/home/andrius/thesis/comparison/antarna/antarna'
    command = ['/usr/bin/python', 'antarna.py','MFE','-Cstr', sequence]
    p = subprocess.Popen(command, cwd=path, stdout=subprocess.PIPE)
    t0, t = time.time(), 0
    while p.poll() is None and t < time_limit:
        t = time.time() - t0

    if p.returncode is not None:
        output = p.communicate()[0].decode('utf-8')
        pr = output.split('\n')[-2].strip()
        return pr
    else:
        p.terminate()
        p.kill()
        return None

def vienna_fold(sequence, time_limit=10):
    command = ['RNAinverse']
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    try:
        output = p.communicate(sequence.encode('utf-8'), timeout=time_limit)
        seq = output[0].decode('utf-8').split()[0].strip()
        return seq
    except:
        p.terminate()
        p.kill()
        
        return None

def learna_fold(sequence, time_limit=10):
    
    if lfold.target != sequence:
        lfold.prep([sequence])

    t0, t = time.time(), 0
    while t < time_limit:
        nucl = lfold.run()
        t = time.time() - t0
        # if nucl[1] == 0:
        if nucl[2] is not None:
            return nucl[2]
        else:
            return nucl[0]
    # return None

def mcts_fold(sequence, time_limit=10):
    nucl = mf(sequence, time_limit=time_limit)
    return nucl

def rlfold_fold(sequence, time_limit=10):
    
    if wrapper.target != sequence:
        print('prepping...')
        wrapper.prep(sequence, permute=True, verbose=False)
    solutions = wrapper.single_fold()
    return solutions

def mrlfold_fold(sequence, time_limit=10):

    if mwrapper.target != sequence:
        print('prepping...')
        mwrapper.prep(sequence, permute=True, verbose=False)
    solutions = mwrapper.single_fold()
    return solutions

def brlfold_fold(sequence, time_limit=10):

    if bwrapper.target != sequence:
        print('prepping...')
        bwrapper.prep(sequence, permute=True, verbose=False)
    solutions = bwrapper.single_fold()
    return solutions

def mass_fold(seq, time_limit=10, file=None, log=None):
    
    config =  { 'encoding_type': 2,
                'use_nucleotides': True,
                'use_padding': True,
                'full_state': True,
                'kernel_size': 45,
                'permute': False,
                'permutation_budget': 50,
                'permutation_radius': 1,
                'permutation_threshold': 5,
                'allow_gu_permutations': False,
                'mutation_probability': 0.5,
                'detailed_comparison': True,
                'reward_exp': 9,
                'write_threshold': 0}

    print('\n\nSequence: {}   |   l={}   |   t={}'.format(seq, len(seq), time_limit))
    # Reward

    # names = ['rlfold', 'mrlfold', 'brlfold']
    fns = [rlfold_fold, mrlfold_fold, brlfold_fold]
    fns = [vienna_fold, modena_fold, nupack_fold, antarna_fold, mcts_fold, learna_fold, rlfold_fold, mrlfold_fold, brlfold_fold]
    names = ['RNAinverse', 'MODENA', 'NUPACK', 'antaRNA', 'rnaMCTS', 'LEARNA', 'rlfold', 'mrlfold', 'brlfold']
    #'LEARNA',learna_fold
    target = Sequence(seq)

    if file is not None:
        file = open(file, 'a+')
        file.write(','.join([seq, str(time_limit), str(len(seq))])+'\n')
        file.write(','.join(['seq', 'source', 't', 'hd', 'md', 'fe', 'ed', 'attempts'])+'\n')
    if log is not None:
        log = open(log, 'a+')

    def evaluate(fn, seq, time_limit):
        t0 = time.time()
        result = fn(seq, time_limit=time_limit)
        t = time.time() - t0
        return result, t

    solved = np.zeros(len(names))
    times = np.zeros(len(names))
    for i, fn in enumerate(fns):
        print('\nUsing {}'.format(names[i]))
        t_start, t_total, hd, attempts = time.time(), 0, 1000, 0
        best, lhd, to_log = None, 500, '---'
        while hd != 0 and t_total < time_limit:
            print(attempts, end='\r')
            result, t_eval = evaluate(fn, seq, time_limit=time_limit-t_total)
            if result is not None:
                if type(result) is str:
                    solutions = [Solution(
                        target=target, 
                        config=config, 
                        string=result, 
                        time=t_eval, 
                        source=names[i])]
                else:
                    solutions = result
                for solution in solutions:
                    attempts += 1
                    hd = solution.hd
                    if hd <= lhd:
                        lhd = hd
                        best = solution
                        to_log = solution.string
                        result_string = '{},{},{:.3f},{},{:.3f},{:.3f},{:.3f},{}\n'.format(
                            best.string, 
                            names[i], 
                            t_total+t_eval/len(solutions), 
                            hd, 
                            best.md, 
                            best.fe, 
                            best.ensemble_defect, 
                            attempts)

                    if hd == 0:
                        solved[i] += 1
                        break

            t_total += t_eval
            
        if best is not None:
            print('    {} m:{}  t={:.3f}  hd={}  md={:.3f}  fe={:.3f}  ed={:.3f}  attempts={}\n'.format(best.string, names[i], t_total/len(solutions), hd, best.md, best.fe, best.ensemble_defect, attempts))
        else:
            print('    Timed out, no solution.')
            result_string = '{}, {}, {:.3f}\n'.format('---', names[i], t_total)
            print('    Solution:   {}      |   t = {:.2f}/{}s, hd: {}, attempt: {}'.format('---', t_eval, time_limit, hd, attempts))
        os.system("pkill RNAinverse")
        times[i] += t_total
        file.write(result_string)
        log.write('{} {:.3f} {}\n'.format(to_log, t_total, names[i]))
    file.write('\n')
    file.close()
    log.close()
    print(names, '\n', solved)
    return solved, times


def eval_dataset(dataset='eterna', time_limit=30):
    logfile = os.path.join(settings.RESULTS, 'comparison', '{}_{}.csv'.format(dataset, random.randint(1,1000)))
    
    n_seqs=29 if dataset=='rfam_taneda' else 100
    data = Dataset(
        dataset=dataset, 
        start=1, 
        n_seqs=n_seqs, 
        encoding_type=2)

    for target in data.sequences:
        mass_fold(target.seq, time_limit=time_limit, file=logfile)

def eval_sequence(dataset='eterna', seq=1, n=10, time_limit=10, logdir=None):
    logfile = os.path.join(logdir, 'solutions.csv')
    logfile2 = os.path.join(logdir, '{}.log'.format(seq))


    
    n_seqs=29 if dataset=='rfam_taneda' else 100
    data = Dataset(
        dataset=dataset, 
        start=1, 
        n_seqs=n_seqs, 
        encoding_type=2)

    target = data.sequences[seq-1]
    log = open(logfile2, 'a+')
    log.write('dataset: {} #{} n={} t={}\n'.format(dataset, seq, n, time_limit))
    log.write(target.seq+'\n')
    log.close()

    solved = np.zeros(9)
    times = np.zeros(9)

    for i in range(n):
        print('\n', seq)
        s, t = mass_fold(target.seq, time_limit=time_limit, file=logfile, log=logfile2)
        solved += s
        times += t
    print('Total solved: ', solved)
    return solved, times
    
if __name__ == "__main__":
    names = ['RNAinverse', 'MODENA', 'NUPACK', 'antaRNA', 'rnaMCTS', 'LEARNA', 'rlfold', 'mrlfold', 'brlfold']

    n_methods = len(names)
    start = 50
    seqs = 50
    n = 1
    t = 60
    dataset = 'eterna'
    logdir = os.path.join(settings.RESULTS, 'comparison', '{}_{}'.format(dataset, random.randint(1,10000)))
    if not os.path.isdir(logdir):
        os.mkdir(logdir)

    total_solved = np.zeros([seqs, n_methods])
    avg_times = np.zeros([seqs, n_methods])
    for i in range(seqs):
        solved, times = eval_sequence(dataset=dataset, seq=i+start, n=n, time_limit=t, logdir=logdir)
        total_solved[i, :] += solved
        avg_times[i, :] += times / n
    summary = os.path.join(logdir, 'summary.log')
    with open(summary, 'w') as f:
        f.write(''.join(names) + '\n')
        for i in range(seqs):
            line = ' '.join([str(x) for x in total_solved[i, :]])
            f.write(str(i+1)+ ' : ' + line + '\n')
        f.write('Total: ' + ' '.join([str(np.sum(total_solved[:, x])) for x in range(n_methods)]))
    print(names, '\n', total_solved)
    print(np.sum(total_solved, axis=1))
    print(times)


    
