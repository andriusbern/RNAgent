import os, subprocess
import rlfold.settings as settings


def fold(sequence, worker):

    path = os.path.join(settings.RESULTS, 'input{}.rna'.format(worker))
    write_file(sequence, path)
    command = 'RNAfold.exe --infile {} --noPS'.format(path)
    c = subprocess.Popen(command.split(), cwd=settings.VIENNA_DIR, shell=True, stdout=subprocess.PIPE)
    u = c.communicate()[0].decode("utf-8")
    u = u.split('\n')[1].split(' ')
    fold = u[0]
    fe = u[1].split(')')[0].split('(')[1]
    if isinstance(fe, str):
        fe = 0.000
    # output = subprocess.getoutput(vpath)
    # print(output)
    # print(fold, fe)
    return fold, fe

def write_file(seq, path):
    """
    Write a sequence to be read by the RNAfold
    """
    with open(path, 'w') as f:
        f.write(seq)


