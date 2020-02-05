import sys, os, time, yaml, random, re
from rlif.rna import Dataset
from rlif.environments import RnaDesign
from rlif.learning import Trainer, get_parameters
from rlif.settings import ConfigManager as settings
from rlif.rna import set_vienna_params

class RLIF(object):
    """
    Container for trained models
    Has methods for solving single sequences and Dataset objects
    """
    def __init__(self):
        self.models = []
        self.envs = []
        self.n_models = 4
        self.dataset = None
        self.initialize()

    def check_sequence(self, seq):
        """
        Returns:
            [True]  if sequence contains only dots and brackets
            [False] if sequence contains only AUGCT
            [None]  if seq contains any other characters
        """
        dot_bracket_check = re.compile(r'[^.)()]').search # Regex for dotbrackets
        nucl_check = re.compile(r'[^AUGCTaugct]').search
        if not bool(dot_bracket_check(seq)):
            return True
        elif not bool(nucl_check(seq)):
            return False
        else:
            return None

    def initialize(self):
        """
        Initializes the models
        """

        _args = settings.model_args[:self.n_models]
        trainers = [Trainer().load_model(num=args[0], checkpoint=args[1], model_only=True) for args in _args]

        self.models = [trainer.model for trainer in trainers]
        self.envs = [RnaDesign({**trainer.config['environment'], **settings.test_config}) for trainer in trainers] # Merge configs and create envs

        # Enable boosting
        for i in range(self.n_models):

            self.envs[i].boosting, self.envs[i].config['boosting'] = settings.model_args[i][2], settings.model_args[i][2]

    def run_dataset(self, dataset, attempts=10, time_limit=60):
        for env in self.envs:
            env.set_data(dataset)

        solutions = [[] for _ in range(dataset.n_seqs)]

        for i, _ in enumerate(dataset.sequences):
            solution, hd = self.solve(attempts=attempts, time_limit=time_limit)
            print(solution.summary()[0])

            if hd == 0:
                solutions[i] = solution
                _ = [env.next_target_structure() for env in self.envs]
        
        return solutions

    def solve(self, sequence=None, attempts=40, time_limit=60):
        """

        """
        if sequence is not None:
            self.prep(sequence)

        start, t = time.time(), 0
        best_solution, lowest_hd, attempt = None, 100, 1
        while attempt < attempts or t < time_limit:
            for num, _ in enumerate(self.models):
                t = time.time() - start
                print('Attempt: {:3}, t: {:.2f}/{}'.format(attempt, t, time_limit), end='\r')
                solution = self.single_run(model_no=num)
                if solution.hd < lowest_hd:
                    best_solution, lowest_hd = solution, solution.hd
                if solution.hd == 0:
                    print('Solved.')
                    return solution, 0
                attempt += 1
        return best_solution, lowest_hd

    def single_run(self, sequence=None, model_no=None):
        """
        Run the current sequence once
        """

        if model_no is None:
            model_no = random.randint(0, len(self.envs)-1)
        env = self.envs[model_no]
        model = self.models[model_no]
        if sequence is not None:
            env.set_data(sequence)

        done = False
        state = env.reset()
        while not done:
            action = model.predict(state)[0]
            state, _, done, _ = env.step(action)

        return env.solution

    def prep(self, sequence):
        """
        Sets the target sequence for all of the environments
        """
        for env in self.envs:
            env.set_sequence(sequence)

    def single_step(self, *args, **kwargs):
        """
        Generates a single nucleotide for the target secondary structure
        """
        state = self.envs[0].solution.get_state()
        action = self.models[0].predict(state)[0]
        state, _, done, _ = self.envs[0].step(action)

        return self.envs[0].solution, done

    def configure(self, *args, **kwargs):
        """
        Set the global parameters for the solver
        """
        mapping = dict(
            # temp=config_vienna,
            params=set_vienna_params,
            permutation_budget=self._configure)

        for arg, val in kwargs.items():
            function = mapping[arg]
            function(arg=val)

    def _configure(self, **kwargs):
        """
        Change the internal parameters of rlif model
        """
        for arg, val in kwargs.items():
            for env in self.envs:
                try:
                    env.config[arg] = val
                    setattr(env, arg, val)
                except: 
                    pass

if __name__ == "__main__":

    solver = RLIF()
    dataset = Dataset(dataset='eterna', start=1, n_seqs=100)
    solver.run_dataset(dataset)