from rlfold.utils import Dataset
from rlfold.environments import RnaDesign
from rlfold.baselines import SBWrapper, get_parameters
from rlfold.interface import show_rna, create_browser
import os, datetime, sys
import rlfold.settings as settings

class Tester(object):
    """
    A class for validating the model while it'solution training
    Provides:
        1. Detailed summary of the results on test sets
        2. Hyperparameter adjustment during training
    """
    def __init__(self, model, budget=10):
        self.model = model
        self.env = RnaDesign(get_parameters) # self.model.env
        self.budget = budget
        self.dir = self.model._model_dir

    def run_test(self, dataset, budget=100, show=False):
        """
        Runs a test on a selected dataset
        """
        training_set = self.model.get_attr('dataset')[0]
        n_seqs = 29 if dataset == 'rfam_taneda' else 100
        test_set = Dataset(dataset=dataset, n_seqs=n_seqs)

        if show:
            driver = create_browser('display')
        self.model.env.set_attr('dataset', test_set)
        self.model.env.set_attr('randomize', False)
        self.model.env.set_attr('meta_learning', False)
        get_sequence = self.model.env.get_attr('next_target')[0]
        self.model.env.set_attr('current_sequence', 0)

        self.test_state = self.model.env.reset()
        solved = []
        
        for n, _ in enumerate(test_set.sequences):
            print('\n', n)
            get_sequence()
            end = False
            episode = 0
            for iteration in range(budget):
                self.done = [False]
                while True:
                    action, _ = self.model.predict(self.test_state)
                    self.test_state, _, self.done, _ = self.model.env.step(action)
                    solutions = self.model.env.get_attr('prev_solution')
                    
                    for solution in solutions:
                        if solution.hd <= 0:
                            solution.summary(True)
                            solved.append([n, solution, iteration+1, budget])
                            print('Solved sequence: {} in {}/{} iterations...'.format(n, iteration+1, budget))
                            end = True
                            if show:
                                show_rna(solution.target.sequence, solution.string, driver, 0, 'display')
                            break
                        if end:
                            break
                if end: 
                    break
        print('Solved ', len(solved), '/', len(test_set.sequences))

        self.write_test_results(solved, test_set)
        
        self.model.env.set_attr('dataset', training_set)
        self.env.reset()

        return solved
        
    def evaluate(self, budget):
        """
        Run evaluation on test sets and save the model'solution checkpoint
        """
        result1 = self.run_test('rfam_learn_test', self.budget)
        result2 = self.run_test('rfam_taneda', self.budget)
        result3 = self.run_test('rfam_learn_validation', self.budget)
        result4 = self.run_test('eterna', self.budget)

    def write_test_results(self, results, dataset):
        """
        Writes the results of the test in ../results/<dataset>/<date>_<solved>.log
        """
        date = datetime.datetime.now().strftime("%m-%d_%H-%M")
        directory = os.path.join(settings.RESULTS, dataset.dataset)
        if not os.path.isdir(directory): os.makedirs(directory)
        filename = os.path.join(directory, '{}_{}.log'.format(date, len(results)))
        budget = results[0][3]
        with open(filename, 'w') as f:

            msg  = 'Dataset: {}, date: {}, solved {}/{} sequences with {} eval budget.\n'.format(
                    dataset.dataset, date, len(results), dataset.n_seqs, budget)
            # msg += 100 * 
            msg += ''.join(['=']*100) + '\n'
            f.write(msg)    
        
            for result in results:
                lines = result[1].summary()
                for line in lines:
                    f.write(line + '\n')
                f.write('Solved in: {}/{}\n'.format(result[2], budget))

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
    



