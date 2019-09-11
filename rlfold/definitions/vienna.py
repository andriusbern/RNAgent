import os, subprocess
import rlfold.settings as settings

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


