

import os, sys
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import tensorflow as tf
import stable_baselines, gym, rlif
from stable_baselines.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv
import numpy as np
import os, yaml, sys, subprocess, time, datetime, random, copy
# Local
from rlif.rna import DotBracket, Dataset
from .tester import Tester
from tqdm import tqdm
from rlif.settings import ConfigManager as settings
import rlif.environments


class Trainer(object):
    """
    Trainer class with vectorized environments
    """

    def __init__(self, env='', subdir='', model_from_file=None):

        self.config = None
        self.env = None
        self.test_env = None
        self.model = None
        self.current_checkpoint = 0
        self.test_runner = Tester(self)

        self.env_name = env
        self._env_type = None
        # self._env_type = get_env_type(self.env_name)
        self.date = datetime.datetime.now().strftime("%m-%d_%H-%M")
        
        self._env_path = settings.TRAINED_MODELS
        self._model_path = None
        self.target = None
        self.setup()

    def setup(self):
        # Test config
        self.reloaded = False
        self.done = True
        self.test_state = None

    def train(self, steps=None):
        """
        Trains the model based on config and saves checkpoints
        """
        if not self.reloaded:
            self._create_model_dir()
        self._check_env_status()
        try:
            evaluate_every = self.config['testing'].get('evaluate_every')
            if evaluate_every is None: evaluate_every = 1000000
            config = dict(
                total_timesteps=steps if steps is not None else evaluate_every,
                tb_log_name=self._unique,
                reset_num_timesteps=False,
                seed=self.config['main']['seed'])
        
            self.reloaded = True
            for _ in range(self.n_steps//evaluate_every):
                self.model.env.set_attr('testing_mode', False)
                self.model = self.model.learn(**config)
                self.test_runner.evaluate(self.config['testing']['test_timeout'])
                self._save()
                self.current_checkpoint += 1
        except KeyboardInterrupt:
            print('\n\n\nStopped training...\n')
            self._save()
         
    def load_model(self, num=None, config_file=None, latest=False, path=None, checkpoint='', n_envs=1, model_only=False, t_env=False):
        """
        Load a saved model either from the 
        """

        import glob
        self._env_path = settings.TRAINED_MODELS
        
        folder_list = glob.glob(self._env_path + '/*') 
        if latest:
            model_path = max(folder_list, key=os.path.getctime)
        else:
            for folder in folder_list:
                if num is not None:
                    if int(folder.split(settings.delimiter)[-1].split('_')[0]) == num:
                        model_path = folder
                        if not os.path.isfile(os.path.join(model_path, 'model.pkl')):
                            model_path = model_path[:-1] + '1'
                        # print('Model path:', model_path)
                        break
                if path is not None:
                    if folder.split(settings.delimiter)[-1] == path:
                        model_path = folder

        self._model_path = model_path
        self.config = get_parameters(self.env_name, self._model_path, config_name=config_file)
        model_file = os.path.join(model_path, 'model{}.pkl'.format(checkpoint))

        self.config['environment']['path'] = model_path
        self.config['main']['n_workers'] = n_envs

        self.n_steps = self.config['main']['n_steps']
        model_object = getattr(stable_baselines, self.config['main']['model'])
        self._unique = model_path.split(settings.delimiter)[-1]
        
        if not model_only:
            self.create_env(test=t_env)    
        self.model = model_object.load(model_file[:-4]) #, env=self.env, tensorboard_log=self._env_path

        return self

    def create_model(self, config_file=None, dataset=None, config_location=None):
        """
        Creates a new RL Model
        """
        args = dict(env_name=self.env_name)
        if config_file is not None:
            args['config_name'] = config_file
        if config_location is not None:
            args['config_location'] = config_location

        c = self.config = get_parameters(**args)
        self._get_model_dir()
        self.config['environment']['path'] = self._model_path
        self.n_steps = self.config['main']['n_steps']
        self.create_env()

        policy_name = c['main']['policy']
        policy_params = c['policy']
        model_name = c['main']['model']
        model_params = c['model']

        self.policy = self._get_policy(policy_name)
        model_object = getattr(stable_baselines, model_name)

        self.model = model_object(
            self.policy, 
            self.env, 
            policy_kwargs={'params':policy_params},
            tensorboard_log=self._env_path, 
            **model_params)

        return self

    def _get_policy(self, policy_name):
        """
        Returns a corresponding policy object either from stable_baselines or the
        custom ones defined in 'rusher/baselines/CustomPolicies.py'
        """
        if hasattr(stable_baselines.common.policies, policy_name):
            return getattr(stable_baselines.common.policies, policy_name)
        else:
            return getattr(rlif.learning, policy_name)

    def create_env(self, test=False):
        """
        Parses the environment to correctly return the attributes based on the spec and type
        Creates a corresponding vectorized environment
        """
        defaults = get_parameters('RnaDesign')['environment']
        config = {**defaults,**self.config['environment']}
        self.env = create_env(self.env_name, config, n_workers=self.config['main']['n_workers'])
        if test:
            self.test_env = create_env(self.env_name, config, n_workers=self.config['main']['n_workers'])

    # Directory management
    def _create_model_dir(self):
        """
        Creates a unique subfolder in the environment directory for the current trained model
        """
        # Create the environment specific directory if it does not exist
        if not os.path.isdir(self._env_path):
            os.makedirs(self._env_path)
        os.makedirs(self._model_path, exist_ok=True)

    def _get_model_dir(self):
        """
        Creates a unique stamp and directory for the model
        """
        try:
            num = max([int(x.split('_')[0]) for x in os.listdir(self._env_path)])# Find the highest id number of current trained models
        except:
            num = 0

        c = self.config['main']
        ce = self.config['environment']
        try:
            dir_name = "{}_{}_buffer{}_Vienna{}_{}_1".format(c['model'], c['n_workers'], ce['buffer_size'], ce['vienna_params'], self.date) # Unique stamp
        except:
            dir_name = self.date
        self._unique = str(num + 1) + '_' + dir_name # Unique identifier of this model
        self._model_path = os.path.join(self._env_path, self._unique) # trained_models/env_type/env/trainID_uniquestamp

    def _delete_incomplete_models(self):
        """
        Deletes directories that do not have the model file saved in them
        """
        import shutil
        count = 0
        for model_folder in os.listdir(self._env_path):
            path = os.path.join(self._env_path, model_folder)
            files = os.listdir(path)
            if 'model.pkl' not in files:
                shutil.rmtree(path)
                count += 1
        print('Cleaned directory {} and removed {} folders.'.format(self._env_path, count))

    def _save(self):
        self.model.save(os.path.join(self._model_path, 'model{}'.format(self.current_checkpoint)))

        # Save config
        with open(os.path.join(self._model_path, 'config.yml'), 'w') as f:
            yaml.dump(self.config, f, indent=4, sort_keys=False, line_break=' ')

        # Fill rate log
        if 'usher' in self.env_name:
            self._save_env_attribute('fill_log')

    def _save_env_attribute(self, attribute):
        """
        Obtains and saves anvironment specific atributes
        (Only one of the environments)
        """
        try:
            data = self.env.get_attr(attribute)
            with open(os.path.join(self._model_path, attribute + '.log'), 'w') as f:            
                for item in data:
                    f.write('%f\n' % item[0])
        except:
            print('Attribute does not exist.')

    def _tensorboard(self, env_name=None):
        # Kill current session
        self._tensorboard_kill()

        # Open the dir of the current env
        cmd = 'tensorboard --logdir ' + self._env_path
        print('Launching tensorboard at {}'.format(self._env_path))
        DEVNULL = open(os.devnull, 'wb')
        subprocess.Popen(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL)
        time.sleep(2)
        webbrowser.open_new_tab(url='http://localhost:6006/#scalars&_smoothingWeight=0.995')
    
    def _tensorboard_kill(self):
        print('Closing current session of tensorboard.')
        os.system("pkill tensorboard")

    def _check_env_status(self):
        try:
            self.env.reset()
        except BrokenPipeError:
            self.create_env()
            self.model.set_env(self.env)
            print('Pipe, Recreating environment')
        except EOFError:
            self.create_env()
            self.model.set_env(self.env)
            print('EOF, Recreating environment')

    def inverse_fold(self, target, solution_count=1, budget=50, permute=True, show=False, verbose=False):
        """
        Method for using the model to generate a nucleotide sequence
        solution given a target dot-bracket sequence
        """
        env = self.test_env
        target = DotBracket(target, 0, 0, encoding_type=self.config['environment']['encoding_type'])
        data = Dataset(sequences=[target])
        env.set_attr('dataset', data)
        env.set_attr('meta_learning', False)
        env.set_attr('permute', permute)
        env.set_attr('ep', 0)
        if verbose==0: env.set_attr('verbose', False)
        env.get_attr('next_target_structure')[0]()
        self.model.set_env(env)
        target = env.get_attr('target_structure')[0]
        if show:
            driver = create_browser('double')
            show_rna(target.seq, 'AUAUAU', driver, 0)
        
        valid_solutions, failed_solutions = [], []
        solution_progress = tqdm(range(solution_count), ncols=90,position=0)
        solution_progress.set_description('Solutions: {:4}/{:4}           '.format(len(valid_solutions), solution_count))
        
        target = self.model.env.get_attr('target_structure')[0]
        print('\nTarget length: {}, Unpaired nucleotides: {:.1f}%, structural motifs: {}'.format(target.len, target.percent_unpaired*100, target.counter),'\n', '='*120)
        try:
            num=0
            for _ in solution_progress:
                attempts_progress = tqdm(range(budget), ncols=90, position=1, leave=False, initial=1)
                attempts_progress.set_description('  Attempt: {:4}/{:4}  HD: {:3}  '.format(0,budget, '   '))
                
                for attempt in attempts_progress:
                    num += 1
                    self.test_state = self.model.env.reset()
                    self.done = [False]
                    end = False

                    while not self.done[0]:
                        action, _ = self.model.predict(self.test_state)
                        self.test_state, _, self.done, _ = self.model.env.step(action)

                    solution = self.model.env.get_attr('prev_solution')[0]
                    if verbose == 2: solution.summary(True)
                    if solution.hd <= 0: 
                        end = True
                        if verbose == 1:
                            solution.summary(True)
                        valid_solutions.append(solution)
                    else:
                        failed_solutions.append(solution)
                    attempts_progress.set_description('  Attempt: {:4}/{:4}  HD: {:3}  '.format(attempt+1,budget, solution.hd))
                    solution_progress.set_description('Solutions: {:4}/{:4}           '.format(len(valid_solutions), solution_count))
                    if show: 
                        show_rna(solution.folded_structure, solution.string, driver, 1)
                        # out = os.path.join(settings.RESULTS, 'images', '{}.jpg'.format(num))
                        # imgkit.from_file(os.path.join(settings.MAIN_DIR, 'display','double.html'), out)
                    if end: break

        except KeyboardInterrupt:
            print('Stopped...')
        return valid_solutions, failed_solutions

    def prep(self, target=None, permute=True, verbose=False, init=False):
        """
        Prepare the model and env for solving a single sequence
        """
        # env = self.test_env
        if target is None:
            
            self.model.env.set_attr('meta_learning', True)
            self.model.env.set_attr('randomize', False)
            self.model.env.set_attr('permute', permute)
            self.model.env.set_attr('ep', 0)
            self.env.set_attr('meta_learning', True)
            self.env.set_attr('randomize', False)
            self.env.set_attr('permute', permute)
            self.env.set_attr('ep', 0)
            self.target = target
            config = self.env.get_attr('config')[0]
            config['permutation_threshold'] = 15
            config['permutation_radius'] = 5
            config['permutation_budget'] = 15
            config['mutation_probability'] = 0.5
            config['allow_gu_permutations'] = True
            if self.env.get_attr('boosting')[0]:
                config['boosting'] = True
            self.env.set_attr('config', config)
            self.model.env.set_attr('config', config)
            if verbose==0: self.model.env.set_attr('verbose', False)
        else:
            target = DotBracket(target, 0, 0, encoding_type=self.config['environment']['encoding_type'])
            if type(target) is not list:
                target = [target]
            data = Dataset(sequences=target)
            self.env.set_attr('dataset', data)
            self.model.env.set_attr('dataset', data)
            
            data_fns = self.model.env.get_attr('set_data')
            for fn in data_fns:
                fn(data)
            nts = self.env.get_attr('next_target_structure')
            for nt in nts:
                nt()

    def single_fold(self):
        """
        Perform a single run on the target
        """
        
        self.test_state = self.model.env.reset()
        self.done = [False]

        while not self.done[0]:
            action, _ = self.model.predict(self.test_state)
            self.test_state, _, self.done, _ = self.model.env.step(action)

        solution = self.model.env.get_attr('prev_solution')
        
        return solution

    def multi_fold(self, target, solution_count=1, budget=50, permute=True, show=False, verbose=False):
        """
        Method for using the model to generate a nucleotide sequence
        solution given a target dot-bracket sequence
        """
        env = self.test_env
        
        target = DotBracket(target, 0, 0, encoding_type=self.config['environment']['encoding_type'])
        data = Dataset(sequences=[target])
        env.set_attr('dataset', data)
        env.set_attr('meta_learning', True)
        env.set_attr('permute', permute)
        env.set_attr('ep', 0)
        if verbose==0: env.set_attr('verbose', False)
        next_structures = env.get_attr('next_target_structure')
        for next_structure in next_structures:
            next_structure()
        [n() for n in next_structures]

        self.model.set_env(env)
        target = env.get_attr('target_structure')[0]
        if show:
            driver = create_browser('double')
            show_rna(target.seq, 'AUAUAU', driver, 0)
        print('\nTarget length: {}, Unpaired nucleotides: {:.1f}%, structural motifs: {}'.format(target.len, target.percent_unpaired*100, target.counter),'\n', '='*120)
        
        valid_solutions, failed_solutions = [], []
        solution_progress = tqdm(range(solution_count), ncols=90,position=0)
        solution_progress.set_description('Solutions: {:4}/{:4}           '.format(len(valid_solutions), solution_count*len(next_structures)))
        
        try:
            for _ in solution_progress:
                attempts_progress = tqdm(range(budget), ncols=90, position=1, leave=False, initial=1)
                attempts_progress.set_description('  Attempt: {:4}/{:4}  HD: {:3}  '.format(0,budget, '   '))
                
                for attempt in attempts_progress:
                    self.test_state = self.model.env.reset()
                    self.done = [False]
                    end = False

                    while not self.done[0]:
                        action, _ = self.model.predict(self.test_state)
                        self.test_state, _, self.done, _ = self.model.env.step(action)

                    solutions = self.model.env.get_attr('prev_solution')
                    for solution in solutions:
                        if verbose == 2: solution.summary(True)
                        if solution.hd <= 0: 
                            end = True
                            if verbose == 1:
                                solution.summary(True)
                            valid_solutions.append(solution)
                        else:
                            failed_solutions.append(solution)
                    attempts_progress.set_description('  Attempt: {:4}/{:4}  HD: {:3}  '.format(attempt+1,budget, solution.hd))
                    solution_progress.set_description('Solutions: {:4}/{:4}           '.format(len(valid_solutions), solution_count*len(next_structures)))
                    if show: show_rna(solution.folded_structure, solution.string, driver, 1)
                    if end: break
                
        except KeyboardInterrupt:
            print('Stopped...')
        return valid_solutions, failed_solutions


    def evaluate_testset(self, dataset='rfam_test', budget=100, permute=False, show=False, pause=0, wait=False):
        """
        Run
        """
        if show:
            driver = create_browser('double')
        self.model.set_env(self.test_env)
        n_seqs=29 if dataset=='rfam_taneda' else 100
        if dataset=='rfam_learn':
            n_seqs=5000
        
        d = Dataset(dataset=dataset, start=1, n_seqs=n_seqs, encoding_type=self.config['environment']['encoding_type'])
        self.model.env.set_attr('dataset', d)
        self.model.env.set_attr('randomize', False)
        self.model.env.set_attr('meta_learning', False)
        get_seq = self.model.env.get_attr('next_target_structure')[0]
        self.model.env.set_attr('current_sequence', 0)
        self.model.env.set_attr('permute', permute)

        self.test_state = self.model.env.reset()
        solved = []
        
        for n, seq in enumerate(d.sequences):
            print(n, '\n')
            get_seq()
            target = self.model.env.get_attr('target_structure')[0]
            if show:
                show_rna(target.seq, 'AUAUAU', driver, 0)
                time.sleep(pause)
            end = False
            ep = 0
            for b in range(budget):
                self.done = [False]
                while not self.done[0]:
                    action, _ = self.model.predict(self.test_state)
                    self.test_state, _, self.done, _ = self.model.env.step(action)
                    solution = self.model.env.get_attr('prev_solution')[0]
                    
                    if self.done[0]:
                        if show and ep%1==0:
                            show_rna(solution.folded_structure, solution.string, driver, 1)
                            
                            time.sleep(pause)
                        if solution.hd <= 0:
                            solution.summary(True)
                            solved.append([n, solution, b+1, budget])
                            print('Solved sequence: {} in {}/{} iterations...'.format(n, b+1, budget))
                            end = True
                            ep += 1
                        if wait: input()
                if end: break
        print('Solved ', len(solved), '/', len(d.sequences))

        return solved

def get_env_type(env_name):
    """
    Get the type of environment from the env_name string
    """

    if 'Rna' in env_name:
        return 'rna'
    else:
        try:
            gym.make(env_name)
            return 'gym'
        except:
            return None

def create_env(env_name, config=None, n_workers=1, image_based=True, **kwargs):
    """
    Parses the environment to correctly return the attributes based on the spec and type
    Creates a corresponding vectorized environment
    """

    def make_rna(rank, **kwargs):
        def _init():
            env_obj = getattr(rlif.environments, env_name)
            env = env_obj(config, rank)
            return env
        return _init
 
    def make_gym(rank, seed=0, **kwargs):
        """
        Decorator for gym environments
        """
        def _init():
            env = gym.make(env_name)
            env.seed(seed + rank)
            return env
        return _init
    
    if config is not None:
        try:
            n_workers = config['main']['n_workers']
        except:
            pass
    mapping = {'gym': make_gym, 'rna':make_rna}
    env_type = get_env_type(env_name)
    env_decorator = mapping[env_type]
    envs = [env_decorator(rank=x) for x in range(n_workers)]

    # Parallelize
    if n_workers > 1:
        if settings.os == 'linux':
            vectorized = SubprocVecEnv(envs, start_method='fork')
        elif settings.os == 'win32':
            vectorized = SubprocVecEnv(envs, start_method='spawn')
    else:
        vectorized = DummyVecEnv(envs)

    return vectorized

def get_parameters(env_name, model_path=None, config_name=None, config_location=None):
    """
    Method for getting the YAML config file of the RL model, policy and environment
    Get config by prioritizing:
        1. Specific config file: /config/[name].yml
        2. From model'solution directory (in case of loading) /trained_models/_/_/_/parameters.yml
        3. /config/[env_name].yml
        4. /config/[env_type].yml
        5. /config/defaults.yml
    """
    # if env_name is not None:
    env_type = get_env_type(env_name)
    env_params = os.path.join(settings.CONFIG, env_name+'.yml')
    if config_location is not None:
        path = config_location
    else:
        if config_name is not None:
            path = os.path.join(settings.CONFIG, config_name + '.yml')
        elif model_path is not None:
            path = os.path.join(model_path, 'config.yml')
        elif os.path.isfile(env_params):
            path = env_params
        else:
            path = os.path.join(settings.CONFIG, env_tyfpe + '.yml')

    with open(path, 'r') as f:
        config = yaml.load(f)

    # Parse some of the config for saving later
    main = config['main']
    config['policy'] = config['policies'][main['policy']]
    config['model'] = config['models'][main['model']]

    return config

        
if __name__ == "__main__":
    b = Trainer('RnaDesign').create_model()

