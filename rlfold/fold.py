from rlfold.definitions import Dataset
from rlfold.baselines import SBWrapper, get_parameters
import rlfold.environments
import os, sys, argparse, time, re
import rlfold.settings as settings
from rlfold.definitions import colorize_nucleotides, highlight_mismatches
import numpy as np

import RNA
RNA.cvar.uniq_ML = 1
header = '\n' + '='*120 + '\n'

param_files = {
    1: 'rna_turner2004.par',
    2: 'rna_turner1999.par',
    3: 'rna_andronescu2007.par',
    4: 'rna_langdon2018.par',
}


def set_vienna_params(param):
    params = os.path.join(settings.MAIN_DIR, 'utils', param_files[param])
    RNA.read_parameter_file(params)


def sort_by_free_energy(designs):
    FE = np.argsort([design.fe for design in designs])
    return [designs[i] for i in FE]


def sort_by_hd(designs):
    HD = np.argsort([design.hd for design in designs])
    return [designs[i] for i in HD]


def sort_by_attribute(designs, attribute):
    attr = np.argsort([getattr(design, attribute) for design in designs])
    return [designs[i] for i in attr]


def solution_summary(designs, hd=False):
    t, _ = highlight_mismatches(designs[0].target.seq, designs[0].target.seq)
    ruler = '    ' + ''.join(['{:3}  '.format(i) for i in range(0, len(designs[0].target.seq), 5)])
    target = '  T: {}'.format(t)
    for i, design in enumerate(designs):
        if i % 10 == 0:
            print(ruler, '\n', target)
        gcau = design.gcau_content()
        content = '  ||  G:{:.2f} | C:{:.2f} | A:{:.2f} | U:{:.2f} |'.format(
            gcau['G'], gcau['C'], gcau['A'], gcau['U'])
        if hd:
            content += ' HD: {:2}'.format(design.hd)
        print('{:4}: {}   FE: {:.3f}'.format(
            i+1, colorize_nucleotides(design.string), design.fe) + content)


def mismatch_summary(designs):
    target = designs[0].target.seq
    ruler = '    ' + ''.join(['{:3}  '.format(i)
                              for i in range(0, len(target), 5)])
    target_formatted = '  T: {}'.format(target)

    for i, design in enumerate(designs):
        if i % 10 == 0:
            print(ruler, '\n', target_formatted)
        _, mismatches = highlight_mismatches(
            design.target.seq, design.folded_design)
        gcau = design.gcau_content()
        dist = RNA.dist_mountain(target, design.folded_design)
        d4 = RNA.bp_distance(target, design.folded_design)
        content = '  ||  G:{:.2f} | C:{:.2f} | A:{:.2f} | U:{:.2f} |'.format(
            gcau['G'], gcau['C'], gcau['A'], gcau['U'])
        content += ' | HD: {:2}'.format(design.hd)
        content += ' | MD: {:.3f}'.format(dist)
        content += ' | BPD: {:2}'.format(d4)
        print('{:4}: {}   FE: {:.3f}'.format(
            i+1, mismatches, design.fe) + content)
        print('    : {}'.format(colorize_nucleotides(design.string)))


def config_summary(args):
    print(
        header,
        '\nConfiguration: \n',
        'Number of solutions:      %i\n' % args.num_solutions,
        'Attempts per solution:    %i\n' % args.attempts,
        'Model:                    %s\n' % model_dict[args.model],
        'Show structure:           %r\n' % args.show,
        'Display failed sequences: %r\n' % args.failed,
        'Permute:                  %r\n' % args.permute,
        'Vienna parameters:        %s\n' % param_files[args.vienna_config],
        'Verbosity                 %i'   % args.verbosity)


def find_unique(solutions):
    hashes = []
    unique = []
    for solution in solutions:
        hashed = hash(solution.string)
        if hashed not in hashes:
            hashes.append(hashed)
            unique.append(solution)
    return unique

def fold(model, args):
    t0 = time.time()

    if args.multi:
        valid, failed = model.multi_fold(
            target=target,
            solution_count=args.num_solutions,
            budget=args.attempts,
            permute=args.permute,
            show=args.show,
            verbose=args.verbosity)
    else:
        valid, failed = model.inverse_fold(
            target=target,
            solution_count=args.num_solutions,
            budget=args.attempts,
            permute=args.permute,
            show=args.show,
            verbose=args.verbosity)

    unique = find_unique(valid)
    t = time.time() - t0
    attempts = model.model.env.get_attr('ep')[0]
    print(header, 'Solutions found: {}, unique: {}, time taken: {:.2f}s, total attempts: {}, t/s: {:.2f}, a/s: {:.2f}\n'.format(
        len(valid),
        len(unique),
        t,
        attempts,
        len(valid)/t,
        attempts/t))

    # Summaries
    if len(unique) > 0:
        valid_designs = sort_by_attribute(unique, 'fe')
        solution_summary(valid_designs)
    if args.failed and len(failed) > 0:
        print(header, '\nFailed solutions')
        failed_designs = sort_by_hd(failed)
        mismatch_summary(failed_designs)
    input()


def load_model(directory, number, checkpoint=None):
    n_envs = 6 if args.multi else 1
    trained_model = SBWrapper(
        'RnaDesign', directory).load_model(number, checkpoint=checkpoint, n_envs=n_envs)
    return trained_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--timeout', type=int, default=60)
    parser.add_argument('-a', '--attempts', type=int, default=10)
    parser.add_argument('-s', '--show', action='store_true')
    parser.add_argument('-n', '--num_solutions', type=int, default=10)
    parser.add_argument('-v', '--verbosity', type=int, default=0)
    parser.add_argument('-m', '--model', type=str, default='0')
    parser.add_argument('-f', '--failed', action='store_false')
    parser.add_argument('-p', '--permute', action='store_false')
    parser.add_argument('-c', '--vienna_config', type=int, default=1)
    parser.add_argument('--multi', action='store_true')
    args = parser.parse_args()
    parser.print_help()
    
    model_dict = {
        '0': ['experiment4', 2, ''], # Best fo sho, low U
        '1': ['t238', 1, '10'], # 
        '2': ['e238', 2, '11'],
        '3': ['experiment5', 1, '12'],
        '4': ['experiment4', 1, '10'],
        '5': ['long', 1, '56'],
        '6': ['trit', 1, '20'], # bad
        # '2': ['l', 1, '20'], # very bad
        '7': ['l2', 1, '32'],
        '8': ['latest', 1, '39'],
        '9': ['latest', 2, '31'],
        '10': ['experiment6', 2, '19'],
        }

    params = model_dict[args.model]
    trained_model = load_model(*params)

    try:
        os.system('clear')
        while True:
            while True:
                os.system('clear')
                config_summary(args)
                target = input(
                    header + '\n\nEnter the target secondary RNA structure in dot-bracket notation: \n')
                dot_bracket_check = re.compile(r'[^.)()]').search  # Regex for dotbrackets
                nucl_check = re.compile(r'[^AUGC]').search

                if not bool(dot_bracket_check(target)) and len(target) > 0:
                    os.system('clear')
                    config_summary(args)
                    seq, _ = highlight_mismatches(target, target)
                    print(header, '\nTarget:  ' + seq)
                    break
                if not bool(nucl_check(target)) and len(target) > 0:
                    nucleotides = colorize_nucleotides(target)
                    target, _ = RNA.fold(target)
                    config_summary(args)
                    seq, _ = highlight_mismatches(target, target)
                    
                    print(header, '\nTarget:  ' + seq)
                    print('\nSequence: ' + nucleotides)
                    break



                elif target.startswith('n'):
                    args.num_solutions = int(target[1:])
                elif target.startswith('a'):
                    args.attempts = int(target[1:])
                elif target.startswith('m'):
                    args.model = str(target[1:])
                    params = model_dict[args.model]
                    trained_model = load_model(*params)
                elif target.startswith('s'):
                    args.show = not args.show
                elif target.startswith('v'):
                    args.verbosity = int(target[1])
                elif target.startswith('f'):
                    args.failed = not args.failed
                elif target.startswith('p'):
                    args.permute = not args.permute
                elif target.startswith('c'):
                    param_file = int(target[1])
                    args.vienna_config = int(target[1])
                    set_vienna_params(param_file)
                else:
                    print('\nInvalid sequence: {}'.format(target))

            fold(trained_model, args)

    except KeyboardInterrupt:
        print('\nExit.')
