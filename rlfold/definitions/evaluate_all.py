from rlfold.definitions import Dataset, Solution, Sequence
from rlfold.utils import show_rna, create_browser
from rlfold.baselines import SBWrapper

import os, datetime, sys, time
import rlfold.settings as settings
import numpy as np
import pandas as pd
sys.path.append('/home/andrius/thesis/code/comparison/learna/src')
sys.path.append('/home/andrius/thesis/code/comparison')
from mcts import fold as mcts_fold
from learna import Learna_fold
import RNA

class Tester(object):
    """
    A class for validating the model while it's training
    Provides:
        1. Detailed summary of the results on test sets
        2. Hyperparameter adjustment during training
    """
    def __init__(self, wrapper, budget=20):
        self.wrapper = wrapper
        self.budget = budget
        # self.dir = self.wrapper._model_path
        self.test_state = None
        self.done = None

    def evaluate(self, method='rlfold', time_limit=60, verbose=False, permute=False, show=True):
        """
        Run evaluation on test sets and save the model'solution checkpoint
        """
        datasets = ['rfam_learn_test', 'rfam_taneda', 'rfam_learn_validation', 'eterna']
        # datasets = ['rfam_taneda', 'eterna']
        results = [None for dataset in datasets]

        
        for i, dataset in enumerate(datasets):
            try:    
                if method == 'rlfold':
                    results[i], desc = self.rlfold_evaluation(
                        dataset=dataset, 
                        time_limit=time_limit, 
                        verbose=verbose, 
                        permute=permute)

                if method == 'learna':
                    results[i], desc = self.learna_evaluation(
                        dataset=dataset, 
                        time_limit=time_limit, 
                        verbose=verbose)
                if method == 'vienna':
                    results[i], desc = self.vienna_evaluation(
                        dataset=dataset, 
                        time_limit=time_limit, 
                        verbose=verbose)
                if method == 'mcts':
                    results[i], desc = self.mcts_evaluation(
                        dataset=dataset, 
                        time_limit=time_limit, 
                        verbose=verbose)
            except KeyboardInterrupt:
                pass

        path = os.path.join(settings.RESULTS, 'training_tests.csv')
        self.write_csv(results, path, desc, time_limit)

    def timed_evaluation(self, dataset='rfam_learn_test', time_limit=60, permute=False, show=False, pause=0, verbose=True):
        """
        Run timed test on a benchmark dataset

        Looks horrible, not really possible to refactor
        """
        model = self.wrapper.model


        # Set a single threaded environment for testing
        model.set_env(self.wrapper.test_env)

        n_seqs=29 if dataset=='rfam_taneda' else 100
        
        test_set = Dataset(
            dataset=dataset, 
            start=1, 
            n_seqs=n_seqs, 
            encoding_type=self.wrapper.config['environment']['encoding_type'])
        
        # Get and set attributes
        model.env.set_attr('dataset', test_set)
        model.env.set_attr('randomize', False)
        model.env.set_attr('meta_learning', True)
        model.env.set_attr('current_sequence', 0)
        model.env.set_attr('permute', permute)
        data = model.env.get_attr('dataset')[0]
        get_next_target = model.env.get_attr('next_target_structure')

        solved  = []
        t_total = 0
        attempts = np.zeros([n_seqs], dtype=np.uint8)
        min_hd   = np.ones([n_seqs],  dtype=np.uint8) * 500
        t_per_sequence = np.zeros([n_seqs])

        # for next_target in get_next_target:
        #     next_target()
        # model.env.reset()
        
        # get_next_target()
        test_state = model.env.reset()
        try:
            while t_total <= time_limit:
                ep_start = time.time()

                # for next_target in get_next_target:
                #     next_target()

                target = model.env.get_attr('target_structure')[0]
                if show:
                    show_rna(target.seq, 'None', driver, 0)
                    time.sleep(pause)
                
                done = [False]
                while not done[0]:

                    action, _ = model.predict(test_state)
                    test_state, _, done, _ = model.env.step(action)

                solution = model.env.get_attr('prev_solution')[0]
                target_id = solution.target.file_nr - 1
                if show:
                    show_rna(solution.folded_design, solution.string, driver, 1)
                    time.sleep(pause)

                attempts[target_id] += 1
                t_episode = time.time() - ep_start
                t_total += t_episode
                t_per_sequence[target_id] += t_episode
                if solution.hd < min_hd[target_id]:
                    min_hd[target_id] = solution.hd

                if solution.hd <= 0:
                    if verbose:
                        solution.summary(True)
                        print('({}/{}) Solved sequence: {} in {} iterations, {:.2f} seconds...\n'.format(
                            len(solved),
                            n_seqs, 
                            target_id, 
                            attempts[target_id], 
                            t_per_sequence[target_id]))

                    # Remove solved target from targets to be solved
                    data.sequences.remove(solution.target)
                    solved.append(
                        [target_id,
                            solution, 
                            attempts[target_id], 
                            min_hd[target_id], 
                            round(t_per_sequence[target_id],2), 
                            time_limit])

        except KeyboardInterrupt:
            pass

        print('Solved {}/{}'.format(len(solved), n_seqs))

        date = self.date = str(datetime.datetime.now().strftime("%m-%d %H:%M"))
        model.set_env(self.wrapper.env) # Restore env
        test_summary = [date, len(solved), time_limit, self.wrapper._model_path, self.wrapper.current_checkpoint, dataset]
        self.write_test_results(solved, test_set, time_limit)
        self.write_detailed_csv(test_summary, min_hd, t_per_sequence)

        return solved, test_summary

    def rlfold_evaluation(self, dataset='rfam_learn_test', time_limit=60, permute=True, verbose=True):

        model = self.wrapper.model
        model.set_env(self.wrapper.test_env)
        conf = self.wrapper.config['environment']
        n_seqs=29 if dataset=='rfam_taneda' else 100
        test_set = Dataset(
            dataset=dataset, 
            start=1, 
            n_seqs=n_seqs, 
            encoding_type=self.wrapper.config['environment']['encoding_type'])

        # Get and set attributes
        model.env.set_attr('dataset', test_set)
        model.env.set_attr('randomize', False)
        model.env.set_attr('meta_learning', False)
        model.env.set_attr('current_sequence', 0)
        model.env.set_attr('permute', permute)
        data = model.env.get_attr('dataset')[0]
        get_next_target = model.env.get_attr('next_target_structure')[0]

        min_hd = np.ones([n_seqs],  dtype=np.uint8) * 500
        t_per_sequence = np.zeros([n_seqs])
        solved = []
        for i, target in enumerate(test_set.sequences):
            get_next_target()
            test_state = model.env.reset()
            t0 = time.time()
            t, attempts = 0, 0
            while t < time_limit:
                done = [False]
                while not done[0]:
                    action, _ = model.predict(test_state)
                    test_state, _, done, _ = model.env.step(action)

                t_solution = time.time()
                t_finish = time.time() - t_solution
                solution = model.env.get_attr('prev_solution')[0]
                attempts += 1
                t = time.time() - t0
                if solution.hd < min_hd[i]:
                    min_hd[i] = solution.hd

                if solution.hd <= 0:
                    # solved.append(solution)
                    solved.append(
                        [i+1,
                        solution, 
                        attempts, 
                        min_hd[i], 
                        round(t_per_sequence[i],2), 
                        time_limit])
                    if verbose:
                        solution.summary(True)
                        print('({}/{}) Solved sequence: {} in {} iterations, {:.2f} seconds...\n'.format(
                            len(solved),
                            n_seqs, 
                            i+1, 
                            attempts, 
                            t_finish))
                    break
            t_per_sequence[i] = t

        print('Solved {}/{}'.format(len(solved), n_seqs))
        date = self.date = str(datetime.datetime.now().strftime("%m-%d %H:%M"))
        test_summary = [date, len(solved), time_limit, dataset, 'LEARNA']
        self.write_test_results(solved, test_set, time_limit)
        self.write_detailed_csv(test_summary, min_hd, t_per_sequence)

        return solved, test_summary


    def learna_evaluation(self, dataset='rfam_learn_test', time_limit=1, verbose=True):
        lfold = Learna_fold()
        conf = self.wrapper.config['environment']

        n_seqs=29 if dataset=='rfam_taneda' else 100
        test_set = Dataset(
            dataset=dataset, 
            start=1, 
            n_seqs=n_seqs, 
            encoding_type=self.wrapper.config['environment']['encoding_type'])

        min_hd = np.ones([n_seqs],  dtype=np.uint8) * 500
        t_per_sequence = np.zeros([n_seqs])
        solved = []
        for i, target in enumerate(test_set.sequences):
            lfold.prep([target.seq])
            t0 = time.time()
            t, attempts = 0, 0
            while t < time_limit:
                t_solution = time.time()
                nucl = lfold.run()
                t_finish = time.time() - t_solution
                solution = Solution(
                    target=target, 
                    config=conf, 
                    string=nucl[0],
                    time=t_finish, 
                    source='learna')

                attempts += 1
                t = time.time() - t0
                if solution.hd < min_hd[i]:
                    min_hd[i] = solution.hd

                if solution.hd <= 0:
                    # solved.append(solution)
                    solved.append(
                        [i+1,
                        solution, 
                        attempts, 
                        min_hd[i], 
                        round(t_per_sequence[i],2), 
                        time_limit])
                    if verbose:
                        solution.summary(True)
                        print('({}/{}) Solved sequence: {} in {} iterations, {:.2f} seconds...\n'.format(
                            len(solved),
                            n_seqs, 
                            i+1, 
                            attempts, 
                            t_finish))
                    break
            t_per_sequence[i] = t

        print('Solved {}/{}'.format(len(solved), n_seqs))
        date = self.date = str(datetime.datetime.now().strftime("%m-%d %H:%M"))
        test_summary = [date, len(solved), time_limit, dataset, 'LEARNA']
        self.write_test_results(solved, test_set, time_limit)
        self.write_detailed_csv(test_summary, min_hd, t_per_sequence)

        return solved, test_summary


    def vienna_evaluation(self, dataset='rfam_learn_test', time_limit=5, verbose=True):
        conf = self.wrapper.config['environment']
        globalt = time.time()
        n_seqs=29 if dataset=='rfam_taneda' else 100
        test_set = Dataset(
            dataset=dataset, 
            start=1, 
            n_seqs=n_seqs, 
            encoding_type=self.wrapper.config['environment']['encoding_type'])

        min_hd = np.ones([n_seqs],  dtype=np.uint8) * 500
        t_per_sequence = np.zeros([n_seqs])
        solved = []

        try:
            for i, target in enumerate(test_set.sequences):
                print('Sequence ', i+1)
                t0 = time.time()
                t, attempts = 0, 0
                while t < time_limit:
                    t_solution = time.time()
                    nucl = RNA.inverse_fold(None, target.seq)
                    t_finish = time.time() - t_solution
                    solution = Solution(
                        target=target, 
                        config=conf, 
                        string=nucl[0],
                        time=t_finish, 
                        source='learna')

                    attempts += 1
                    t = time.time() - t0
                    if solution.hd < min_hd[i]:
                        min_hd[i] = solution.hd

                    if solution.hd <= 0:
                        solved.append(
                            [i+1,
                            solution, 
                            attempts, 
                            min_hd[i], 
                            round(t_per_sequence[i],2), 
                            time_limit])
                        if verbose:
                            solution.summary(True)
                            print('({}/{}) Solved sequence: {} in {} iterations, {:.2f} seconds...\n'.format(
                                len(solved),
                                n_seqs, 
                                i+1, 
                                attempts, 
                                t_finish))
                        break
                t_per_sequence[i] = t
        except KeyboardInterrupt:
            print('Exit')

        print('Solved {}/{}'.format(len(solved), n_seqs))
        date = self.date = str(datetime.datetime.now().strftime("%m-%d %H:%M"))
        test_summary = [date, len(solved), time_limit, dataset, 'RNAinverse']
        self.write_test_results(solved, test_set, time_limit)
        self.write_detailed_csv(test_summary, min_hd, t_per_sequence)

        return solved, test_summary


    def mcts_evaluation(self, dataset='rfam_learn_test', time_limit=5, verbose=True):
        conf = self.wrapper.config['environment']
        globalt = time.time()
        n_seqs=29 if dataset=='rfam_taneda' else 100
        test_set = Dataset(
            dataset=dataset, 
            start=1, 
            n_seqs=n_seqs, 
            encoding_type=self.wrapper.config['environment']['encoding_type'])

        min_hd = np.ones([n_seqs],  dtype=np.uint8) * 500
        t_per_sequence = np.zeros([n_seqs])
        solved = []

        try:
            for i, target in enumerate(test_set.sequences):
                print('Sequence ', i+1)
                t0 = time.time()
                t, attempts = 0, 0
                while t < time_limit:
                    t_solution = time.time()
                    nucl = RNA.inverse_fold(None, target.seq)
                    nucl = mcts_fold(target.seq, time_limit=time_limit)
                    t_finish = time.time() - t_solution
                    solution = Solution(
                        target=target, 
                        config=conf, 
                        string=nucl,
                        time=t_finish, 
                        source='learna')

                    attempts += 1
                    t = time.time() - t0
                    if solution.hd < min_hd[i]:
                        min_hd[i] = solution.hd

                    if solution.hd <= 0:
                        solved.append(
                            [i+1,
                            solution, 
                            attempts, 
                            min_hd[i], 
                            round(t_per_sequence[i],2), 
                            time_limit])
                        if verbose:
                            solution.summary(True)
                            print('({}/{}) Solved sequence: {} in {} iterations, {:.2f} seconds...\n'.format(
                                len(solved),
                                n_seqs, 
                                i+1, 
                                attempts, 
                                t_finish))
                        break
                t_per_sequence[i] = t
        except KeyboardInterrupt:
            print('Exit')

        print('Solved {}/{}'.format(len(solved), n_seqs))
        date = self.date = str(datetime.datetime.now().strftime("%m-%d %H:%M"))
        test_summary = [date, len(solved), time_limit, dataset, 'RNAinverse']
        self.write_test_results(solved, test_set, time_limit)
        self.write_detailed_csv(test_summary, min_hd, t_per_sequence)

        return solved, test_summary


    def write_test_results(self, results, dataset, time_limit):
        """
        Writes the results of the test in ../results/<dataset>/<date>_<solved>.log
        """
        date = datetime.datetime.now().strftime("%m-%d_%H-%M")
        directory = os.path.join(settings.RESULTS, dataset.dataset)

        config = self.wrapper.config['environment']
        if not os.path.isdir(directory): os.makedirs(directory)
        filename = os.path.join(directory, '{}_{}.log'.format(date, len(results)))
        with open(filename, 'w') as f:

            msg  = 'Dataset: {}, date: {}, solved {}/{} sequences with {}s eval budget.\n'.format(
                    dataset.dataset, date, len(results), dataset.n_seqs, time_limit)
            msg += ''.join(['=']*100) + '\n'

            msg += 'Permutation strategy: {}, budget: {}, threshold: {}, radius: {}\n'.format(
                config['permute'], config['permutation_budget'], config['permutation_threshold'], config['permutation_radius'])
            f.write(msg)    

            for result in results:
                lines = result[1].summary()
                for line in lines:
                    f.write(line + '\n')
                try:
                    f.write('Solved in: {} attempts, {}s\n'.format(result[2], result[4]))
                except:
                    pass

    def write_csv(self, results, path, test_summary, time_limit):
        """
        """
        header = 'Date, Solved, Time(s), Model, Checkpoint, learn_test, taneda, learn_validation, eterna\n'
        data   = ', '.join([str(x) for x in test_summary[:-1]])
        data += ', ' + ', '.join([str(len(x)) for x in results]) + '\n'
        if not os.path.isfile(path):
            with open(path, 'w') as f:
                f.write(header)
        else:
            with open(path, 'a') as f:
                f.write(data)
    
    def write_detailed_csv(self, test_summary, hd, t_per_sequence):
        """
        Write a csv with individual sequence details 
        """
        
        header = 'Date, Solved, Time(s), Model, Checkpoint, Dataset'

        sequence_data = ''
        for i in range(len(hd)):
            sequence_data += ', ' + str(hd[i])
            header += ', seq{}'.format(i+1)
        for i in range(len(t_per_sequence)):
            sequence_data += ', ({})'.format(round(t_per_sequence[i],2))
            header += ', seq{}(t)'.format(i+1)
        header += '\n'

        data_entry = ', '.join([str(x) for x in test_summary]) + sequence_data + '\n'
        dataset = test_summary[-1]
        path = os.path.join(settings.RESULTS, dataset+'.csv')
        if not os.path.isfile(path):
            with open(path, 'w') as f:
                f.write(header)
        with open(path, 'a') as f:
            f.write(data_entry)

    def write_test_summary(self):
        """
        Detailed statistics on the tests
        2. General overview of progress during training
        """

    def modify_parameters(self):
        """
        Change the model'solution hyperparameters in between checkpoints
        """

if __name__ == "__main__":

    wrapper = SBWrapper(
        'RnaDesign', 'experiment4').load_model(2, checkpoint='', n_envs=1)
    tester = Tester(wrapper)

    # solved, summary = tester.learna_evaluation()
    tester.evaluate('mcts', time_limit=5, verbose=True, permute=True)
    



