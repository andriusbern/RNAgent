from rlfold.utils import Dataset
from rlfold.interface import show_rna, create_browser
import os, datetime, sys, time
import rlfold.settings as settings
import numpy as np
import pandas as pd

class Tester(object):
    """
    A class for validating the model while it'solution training
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

    def timed_evaluation(self, dataset='rfam_learn_test', time_limit=60, permute=False, show=False, pause=0, verbose=True):
        """
        Run timed test on a dataset
        """
        model = self.wrapper.model
        model.set_env(self.wrapper.test_env)

        if show:
            driver = create_browser('double')

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
        get_seq = model.env.get_attr('next_target')[0]

        self.test_state = model.env.reset()
        solved  = []
        t_total = 0
        attempts = np.zeros([n_seqs], dtype=np.uint8)
        min_hd   = np.ones([n_seqs], dtype=np.uint8) * 500
        time_taken = np.zeros([n_seqs])

        try:
            while t_total <= time_limit:
                ep_start = time.time()
                get_seq()
                target = model.env.get_attr('target')[0]
                if show:
                    show_rna(target.seq, 'AUAUAU', driver, 0)
                    time.sleep(pause)
                episode = 0
                
                self.done = [False]
                while not self.done[0]:

                    action, _ = model.predict(self.test_state)
                    self.test_state, _, self.done, _ = model.env.step(action)
                    solution = model.env.get_attr('prev_solution')[0]
                    num = solution.target.file_nr - 1

                    if self.done[0]:

                        if show and episode%1==0:
                            show_rna(solution.folded_design.seq, solution.string, driver, 1)
                            time.sleep(pause)

                        if solution.hd < min_hd[num]: min_hd[num] = solution.hd
                        if solution.hd <= 0:
                            data = model.env.get_attr('dataset')[0]
                            data.sequences.remove(solution.target)
                            episode += 1

                        attempts[num] += 1
                        t_episode = time.time() - ep_start
                        t_total += t_episode
                        time_taken[num] += t_episode

                        if solution.hd <= 0:
                            if verbose:
                                solution.summary(True)
                                print('({}/{}) Solved sequence: {} in {} iterations, {:.2f} seconds...\n'.format(len(solved), n_seqs, num, attempts[num], time_taken[num]))
                            solved.append([num, solution, attempts[num], min_hd[num], round(time_taken[num],2), time_limit])
        except KeyboardInterrupt:
            pass
        print('Solved {}/{}'.format(len(solved), n_seqs))

        date = self.date = str(datetime.datetime.now().strftime("%m-%d %H:%M"))
        model.set_env(self.wrapper.env) # Restore env
        description = [date, len(solved), time_limit, self.wrapper._model_path, self.wrapper.current_checkpoint, dataset]
        self.write_test_results(solved, test_set, time_limit)
        self.write_detailed_csv(description, min_hd, time_taken)

        return solved, description
        
    def evaluate(self, time_limit=60, verbose=False, permute=False):
        """
        Run evaluation on test sets and save the model'solution checkpoint
        """
        r1, desc = self.timed_evaluation('rfam_learn_test', time_limit, verbose=verbose, permute=permute)
        r2, _    = self.timed_evaluation('rfam_taneda', time_limit, verbose=verbose, permute=permute)
        r3, _    = self.timed_evaluation('rfam_learn_validation', time_limit, verbose=verbose, permute=permute)
        r4, _    = self.timed_evaluation('eterna', time_limit, verbose=verbose, permute=permute)

        path = os.path.join(settings.RESULTS, 'training_tests.csv')
        self.write_csv([r1, r2, r3, r4], path, desc, time_limit)

    def write_test_results(self, results, dataset, time_limit):
        """
        Writes the results of the test in ../results/<dataset>/<date>_<solved>.log
        """
        date = datetime.datetime.now().strftime("%m-%d_%H-%M")
        directory = os.path.join(settings.RESULTS, dataset.dataset)
        if not os.path.isdir(directory): os.makedirs(directory)
        filename = os.path.join(directory, '{}_{}.log'.format(date, len(results)))
        with open(filename, 'w') as f:

            msg  = 'Dataset: {}, date: {}, solved {}/{} sequences with {}s eval budget.\n'.format(
                    dataset.dataset, date, len(results), dataset.n_seqs, time_limit)
            # msg += 100 * 
            msg += ''.join(['=']*100) + '\n'
            f.write(msg)    

            for result in results:
                lines = result[1].summary()
                for line in lines:
                    f.write(line + '\n')
                try:
                    f.write('Solved in: {} attempts, {}s\n'.format(result[2], result[4]))
                except:
                    pass
    

    def write_csv(self, results, path, description, time_limit):
        """
        """
        header = 'Date, Solved, Time(s), Model, Checkpoint, learn_test, taneda, learn_validation, eterna\n'
        data   = ', '.join([str(x) for x in description[:-1]])
        data += ', ' + ', '.join([str(len(x)) for x in results]) + '\n'
        if not os.path.isfile(path):
            with open(path, 'w') as f:
                f.write(header)
        else:
            with open(path, 'a') as f:
                f.write(data)
    
    def write_detailed_csv(self, description, hd, time_taken):
        """
        Write a csv with individual sequence details 
        """
        
        header = 'Date, Solved, Time(s), Model, Checkpoint, Dataset'

        sequence_data = ''
        for i in range(len(hd)):
            sequence_data += ', ' + str(hd[i])
            header += ', seq{}'.format(i+1)
        for i in range(len(time_taken)):
            sequence_data += ', ({})'.format(round(time_taken[i],2))
            header += ', seq{}(t)'.format(i+1)
        header += '\n'

        data_entry = ', '.join([str(x) for x in description]) + sequence_data + '\n'
        dataset = description[-1]
        path = os.path.join(settings.RESULTS, dataset+'.csv')
        if not os.path.isfile(path):
            with open(path, 'w') as f:
                f.write(header)
        with open(path, 'a') as f:
            f.write(data_entry)

    def save(self):
        """
        Saves the current model
        """

    def write_test_summary(self):
        """
        Detailed statistics on the tests
        2. General overview of progress during training
        """

    def modify_parameters(self):
        """
        Change the model'solution hyperparameters in between checkpoints
        """
    



