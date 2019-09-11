from rlfold.definitions import Dataset
from rlfold.interface import show_rna, create_browser
import os, datetime, sys, time
import rlfold.settings as settings
import numpy as np
import pandas as pd

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

    def evaluate(self, time_limit=60, verbose=False, permute=False, show=False, pause=0):
        """
        Run evaluation on test sets and save the model'solution checkpoint
        """
        datasets = ['rfam_learn_test', 'rfam_taneda', 'rfam_learn_validation', 'eterna']
        # datasets = ['rfam_taneda', 'eterna']
        results = [None for dataset in datasets]

        for i, dataset in enumerate(datasets):
            results[i], desc = self.timed_evaluation(
                dataset=dataset, 
                time_limit=time_limit, 
                verbose=verbose, 
                permute=permute, 
                show=show, 
                pause=pause)

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
        model.env.set_attr('meta_learning', False)
        model.env.set_attr('current_sequence', 0)
        model.env.set_attr('permute', permute)
        data = model.env.get_attr('dataset')[0]
        get_next_target = model.env.get_attr('next_target_structure')

        if show:
            driver = create_browser('double')


        solved  = []
        t_total = 0
        attempts = np.zeros([n_seqs], dtype=np.uint8)
        min_hd   = np.ones([n_seqs],  dtype=np.uint8) * 500
        t_per_sequence = np.zeros([n_seqs])

        for next_target in get_next_target:
            next_target()
        model.env.reset()
        
        # get_next_target()
        try:
            test_state = model.env.reset()
            while t_total <= time_limit:
                ep_start = time.time()

                for next_target in get_next_target:
                    next_target()

                target = model.env.get_attr('target_structure')[0]
                if show:
                    show_rna(target.seq, 'None', driver, 0)
                    time.sleep(pause)
                
                done = [False]
                while not done[0]:

                    action, _ = model.predict(test_state)
                    test_state, _, done, _ = model.env.step(action)
                    # print(done)
                    solution = model.env.get_attr('prev_solution')[0]
                    target_id = solution.target.file_nr - 1

                    # if done[0]:
                        # Display solution
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
    



