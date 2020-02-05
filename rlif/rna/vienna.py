import os, subprocess, sys
from rlif.settings import ConfigManager as settings

import RNA
import rlif

def set_vienna_params(params):
    """
    Set the energy parameters of the RNAfold
    """
    param_files = {
        1: 'rna_turner2004.par',
        2: 'rna_turner1999.par',
        3: 'rna_andronescu2007.par',
        4: 'rna_langdon2018.par'}

    params = os.path.join(settings.MAIN_DIR, 'utils', 'parameters', param_files[param])
    RNA.read_parameter_file(params)
    return params

def config_vienna(**kwargs):
    """
    Configure RNAfold
    """
    for arg, val in kwargs.items():
        setattr(RNA.cvar, arg, val)

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
        free_energy = 0.
    
    return secondary_structure, free_energy

