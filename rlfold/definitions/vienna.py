import os, subprocess
import rlfold.settings as settings
try:
    sys.path.remove('/usr/local/lib/python3.6/site-packages')
except:
    pass
import RNA

def set_vienna_params(param):
    param_files = {
        1: 'rna_turner2004.par',
        2: 'rna_turner1999.par',
        3: 'rna_andronescu2007.par',
        4: 'rna_langdon2018.par'}

    params = os.path.join(settings.MAIN_DIR, 'utils', param_files[param])
    RNA.read_parameter_file(params)


def fold(sequence, worker):
    """
    For Windows 
    Call RNAfold.exe using subprocess and return 
    the folded secondary structure sequence and its free energy
    """
    path = os.path.join(settings.RESULTS, 'input{}.rna'.format(worker))
    write_file(sequence, path)
    command = 'RNAfold.exe --infile {} --noPS'.format(path)
    output = subprocess.Popen(command.split(), cwd=settings.VIENNA_DIR, shell=True, stdout=subprocess.PIPE)
    message = output.communicate()[0].decode("utf-8")
    message = message.split('\n')[1].split(' ')
    secondary_structure = message[0]
    free_energy = message[1].split(')')[0].split('(')[1]
    try:
        free_energy = float(free_energy)
    except:
        free_energy = 0.000
    
    return secondary_structure, free_energy

def write_file(seq, path):
    """
    Write a sequence to be read by the RNAfold
    """
    with open(path, 'w') as f:
        f.write(seq)

def zukersubopt(seq):
    """
    Compute Zuker type suboptimal structures.
    Compute Suboptimal structures according to M. Zuker, i.e. for every 
    possible base pair the minimum energy structure containing the resp.
    base pair. Returns a list of these structures and their energies.
    """
    seqs = RNA.zukersubopt(seq)
    for sub in range(seqs.size()):
        subseq = seqs.get(sub)
        print(sub, subseq.structure, subseq.energy)
        

def boltzmann_sampling(seq):
    md = RNA.md()
    md.uniq_ML = 1
    fc = RNA.fold_compound(sequence, md)
    (dot_bracket, mfe) = fc.mfe()
    fc.exp_params_rescale(mfe)
    fc.pf()
    ensemble = []
    i = fc.pbacktrack(100, store_structure, ensemble, RNA.PBACKTRACK_NON_RD)

    for structure in ensemble:
        print(structure)


def probability(dot_br):
    RT = 1.98717 * (273.15 + temperature) / 1000
    md = RNA.md()
    md.temperature = temperature
    fc = RNA.fold_compound(sequence, md)
    pf_structure, pf_energy = fc.pf()
    energy = fc.eval_structure(structure)

    return math.exp(-1*(energy - pf_energy)/RT)


def sample():
    dg = rbp.DependencyGraphMT(lo)