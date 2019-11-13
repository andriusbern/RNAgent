import sys, os, time, yaml, random
from rlif.rna import Dataset
from rlif.environments import RnaDesign
from rlif.learning import Trainer, get_parameters
from rlif.settings import ConfigManager as settings


class RLIF(object):
    """
    Container for best trained models

    Has methods for solving single sequences and Dataset objects
    """
    def __init__(self):
        self.models = []
        self.envs = []
        self.dataset = None
        self.initialize()

    def initialize(self):
        """
        Initializes the models
        """
        trainers = [Trainer().load_model(2, checkpoint='8', n_envs=1, model_only=True),
                    Trainer().load_model(2, checkpoint='8', n_envs=1, model_only=True),
                    # Trainer().load_model(3, checkpoint='9', n_envs=1, model_only=True),
                    Trainer().load_model(3, checkpoint='9', n_envs=1, model_only=True)]
                    # Trainer().load_model(1, checkpoint='10', n_envs=1, model_only=True),
                    # Trainer().load_model(1, checkpoint='10', n_envs=1, model_only=True),
                    # Trainer().load_model(1, checkpoint='12', n_envs=1, model_only=True)]
                    # Trainer().load_model(4, checkpoint='9', n_envs=1, model_only=True),
                    # Trainer().load_model(5, checkpoint='11', n_envs=1, model_only=True)]

        with open(os.path.join(settings.CONFIG, 'Testing.yml'), 'r') as f:
            test_config = yaml.load(f)['environment']

        self.models = [trainer.model for trainer in trainers]
        self.envs = [RnaDesign({**t.config['environment'], **test_config}) for t in trainers]
        self.envs[0].boosting, self.envs[0].config['boosting'] = True, True
    
    def initialize2(self):
        """
        Initializes the models
        """
        trainers = [
            Trainer('RnaDesign', 'experiment4').load_model(2, checkpoint='8', n_envs=4),
            Trainer('RnaDesign', 'experiment4').load_model(2, checkpoint='8', n_envs=4),
            Trainer('RnaDesign', 'experiment4').load_model(1, checkpoint='10', n_envs=4),
            Trainer('RnaDesign', 'experiment4').load_model(1, checkpoint='10', n_envs=4)]

        with open(os.path.join(settings.CONFIG, 'Testing.yml'), 'r') as f:
            test_config = yaml.load(f)['environment']

        self.models = [trainer.model for trainer in trainers]
        self.envs = [RnaDesign({**t.config['environment'], **test_config}) for t in trainers]
        self.envs[0].boosting, self.envs[0].config['boosting'] = True, True
        # self.envs[2].boosting, self.envs[2].config['boosting'] = True, True
        
    def load_dataset(self, dataset):
        self.dataset = None

    def run_dataset(self, dataset, attempts=10, time_limit=60):
        for env in self.envs:
            env.set_data(dataset)

        solutions = [[] for _ in range(dataset.n_seqs)]

        for i, target in enumerate(dataset.sequences):
            solution, hd = self.solve(attempts=attempts, time_limit=time_limit)
            print(solution.summary()[0])

            if hd == 0:
                solutions[i] = solution
            for env in self.envs:
                env.next_target_structure()
        
        return solutions
            
    def solve(self, sequence=None, attempts=40, time_limit=60):
        """

        """
        if sequence is not None:
            [env.set_data(sequence) for env in self.envs]

        start, t = time.time(), 0
        best, lhd, attempt = None, 100, 0
        while attempt < attempts or t < time_limit:
            for num, model in enumerate(self.models):
                t = time.time() - start
                print('Attempt: {:3}, t: {:.2f}/{}'.format(attempt+1, t, time_limit), end='\r')
                env = self.envs[num]
                done = False
                states = env.reset()
                while not done:
                    actions = model.predict(states)[0]
                    states, _, done, _ = env.step(actions)
                solution = env.solution
                if solution.hd < lhd:
                    best, lhd = solution, solution.hd
                if solution.hd == 0:
                    return solution, 0
                attempt += 1
        return best, lhd

    def single_run(self, sequence=None):
        
        index = random.randint(0, len(self.envs)-1)
        env = self.envs[index]
        model = self.models[index]

        if sequence is not None:
            env.set_data(sequence)

        done = False
        state = env.reset()
        while not done:
            action = model.predict(state)[0]
            state, _, done, _ = env.step(action)
        return env.solution

    def prep(self, sequence):
        for env in self.envs:
            env.set_sequence(sequence)

    def single_step(self, sequence=None):
        
        # index = random.randint(0, len(self.envs)-1)
        env = self.envs[0]
        model = self.models[0]
        state = env.solution.get_state()
        # while not done:
        action = model.predict(state)[0]
        state, _, done, _ = env.step(action)
        # return env.solution
        return env.solution, done

    def prep(self, sequence):
        for env in self.envs:
            env.set_sequence(sequence)

if __name__ == "__main__":

    solver = RLIF()
    dataset = Dataset(dataset='eterna', start=1, n_seqs=100)
    solver.run_dataset(dataset)