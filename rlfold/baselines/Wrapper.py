import rlfold.settings as settings
from stable_baselines.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv
from stable_baselines.common.schedules import LinearSchedule, linear_interpolation
from collections import deque
import numpy as np
import os, yaml, sys, subprocess, webbrowser, time, datetime, random, copy
import matplotlib.pyplot as plt
import stable_baselines, gym, rlfold #,# pybullet_envs, rusher, nao_rl
from stable_baselines import PPO2, GAIL
import cv2
from rlfold.interface import show_rna, create_browser
import rlfold.environments
# Local
from rlfold.utils import Sequence, Dataset, Tester
import rlfold.settings as settings

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.info('TensorFlow')
tf.logging.set_verbosity(tf.logging.ERROR)
tf.logging.info('TensorFlow')
import multiprocessing as mp
# mp.set_start_method('spawn')

def get_env_type(env_name):
    """
    Get the type of environment from the env_name string
    """
    if 'usher' in env_name:
        return 'binpacking'
    elif 'Bullet' in env_name:
        return 'bullet'
    elif 'Nao' in env_name or 'Quadruped' in env_name:
        return 'vrep'

    elif 'Rna' in env_name:
        return 'rna'
    else:
        try:
            gym.make(env_name)
            return 'gym'
        except:
            print('{} not found.'.format(env_name))
            return None

def create_env(env_name, config=None, n_workers=1, image_based=True, **kwargs):
    """
    Parses the environment to correctly return the attributes based on the spec and type
    Creates a corresponding vectorized environment
    """

    def make_rna(rank, **kwargs):
        def _init():
            env_obj = getattr(rlfold.environments, env_name)
            env = env_obj(config, rank)
            return env
        return _init

    def make_vrep(**kwargs):
        """
        Decorator for vrep environments
        """
        def _init():
            env = nao_rl.make(env_name)
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
        n_workers = config['main']['n_workers']
    mapping = {'vrep': make_vrep, 'gym': make_gym, 'rna':make_rna}
    env_type = get_env_type(env_name)
    env_decorator = mapping[env_type]
    envs = [env_decorator(rank=x) for x in range(n_workers)]

    # Parallelize
    if n_workers > 1:
        if settings.os == 'linux':
            vectorized = SubprocVecEnv(envs, start_method='forkserver')
        elif settings.os == 'win32':
            vectorized = SubprocVecEnv(envs, start_method='spawn')
    else:
        # Non multi-processing env
        vectorized = DummyVecEnv(envs)
    # Enable frame stacking in the environment
    if 'frame_stack' in config['main'].keys():
        if config['main']['frame_stack'] != 0:
            vectorized = VecFrameStack(vectorized, n_stack=config['main']['frame_stack'])

    return vectorized


def get_parameters(env_name, model_path=None, config_name=None, config_location=None):
    """
    Method for getting the YAML config file of the RL model, policy and environment
    Get config by prioritizing:
        1. Specific config file: /config/[name].yml
        2. From model's directory (in case of loading) /trained_models/_/_/_/parameters.yml
        3. /config/[env_name].yml
        4. /config/[env_type].yml
        5. /config/defaults.yml
    """

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
            path = os.path.join(settings.CONFIG, env_type + '.yml')

    with open(path, 'r') as f:
        config = yaml.load(f)
    print('\nLoaded config file from: {}\n'.format(path))

    # Parse some of the config for saving later
    main = config['main']
    config['policy'] = config['policies'][main['policy']]
    config['model'] = config['models'][main['model']]

    return config

class SBWrapper(object):
    """
    Wrapper for stable_baselines library
    """

    def __init__(self, env, subdir='', model_from_file=None):

        self.config = None
        self.env = None
        self.model = None
        self.current_checkpoint = 0
        self.test_runner = Tester(self)

        self.env_name = env
        self._env_type = get_env_type(self.env_name)
        self.date = datetime.datetime.now().strftime("%m-%d_%H-%M")
        
        self._env_path = os.path.join(settings.TRAINED_MODELS, self._env_type, env, subdir)
        self._model_path = None
        print('Loading path {}'.format(self._env_path))

        self.setup()

    def setup(self):
        # Test config
        self.reloaded = False
        self.done = True
        self.test_state = None
         
    def load_model(self, num=None, config_file=None, latest=False, path=None):
        """
        Load a saved model either from the 
        """

        import glob
        assert os.path.isdir(self._env_path), 'Path {} does not exist.'.format(self._env_path)

        folder_list = glob.glob(self._env_path + '/*') 
        if latest:
            model_path = max(folder_list, key=os.path.getctime)
        else:
            for folder in folder_list:
                if num is not None:
                    print(folder.split(settings.delimiter))
                    if int(folder.split(settings.delimiter)[-1].split('_')[0]) == num:
                        model_path = folder
                        if not os.path.isfile(os.path.join(model_path, 'model.pkl')):
                            model_path = model_path[:-1] + '1'
                        print('Model path:', model_path)
                        break
                if path is not None:
                    print(folder)
                    print(folder.split(settings.delimiter)[-1])
                    if folder.split(settings.delimiter)[-1] == path:
                        model_path = folder


        self._model_path = model_path
        self.config = get_parameters(self.env_name, self._model_path, config_name=config_file)
        self.config['environment']['path'] = model_path

        self.n_steps = self.config['main']['n_steps']
        model_file = os.path.join(model_path, 'model.pkl')
        model_object = getattr(stable_baselines, self.config['main']['model'])
        self._unique = model_path.split(settings.delimiter)[-1]
        print('Unique: {}'.format(self._unique))
        
        self.create_env()
        self.model = model_object.load(model_file[:-4], env=self.env, tensorboard_log=self._env_path)
        self.reloaded = True
        print('Loading file {}'.format(model_file))

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
        print('\nCreating {} model...'.format(model_name))

        self.policy = self._get_policy(policy_name)
        model_object = getattr(stable_baselines, model_name)

        if 'DDPG' in model_name:
            from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec, NormalActionNoise
            n_actions = self.env.action_space.shape[0]
            model_params['action_noise'] = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
        
        if 'Custom' in policy_name:
            if 'GAIL' in model_name:
                from stable_baselines.gail import ExpertDataset
                data = ExpertDataset(expert_path=dataset, traj_limitation=-1, verbose=1)
                self.env = self.test_env
                self.model = model_object(self.policy, self.env, expert_dataset=data, policy_kwargs={'params':policy_params},
                                    tensorboard_log=self._env_path, **model_params)

            else:
                self.model = model_object(self.policy, self.env, policy_kwargs={'params':policy_params},
                                    tensorboard_log=self._env_path, **model_params)

        else:
            self.model = model_object(self.policy, self.env, tensorboard_log=self._env_path, **model_params)

        

        return self

    def _get_policy(self, policy_name):
        """
        Returns a corresponding policy object either from stable_baselines or the
        custom ones defined in 'rusher/baselines/CustomPolicies.py'
        """
        if hasattr(stable_baselines.common.policies, policy_name):
            return getattr(stable_baselines.common.policies, policy_name)
        else:
            return getattr(rlfold.baselines, policy_name)

    def create_env(self):
        """
        Parses the environment to correctly return the attributes based on the spec and type
        Creates a corresponding vectorized environment
        """
        print('Creating {} Environment...\n'.format(self.env_name))

        if self.config['main']['model'] == 'PPO1': self.config['main']['n_workers'] = 1
        # else: n_workers = self.config['main']['n_workers']
        self.env = create_env(self.env_name, self.config)
        testconf = copy.deepcopy(self.config)
        testconf['environment']['meta_learning'] = False
        testconf['main']['n_workers'] = 1
        # if self._env_type is not 'vrep' or self._env_type is not 'rna':
        self.test_env = create_env(self.env_name, testconf, 1)

    from stable_baselines.common.policies import CnnLnLstmPolicy

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
        dir_name = "{}_{}_SL{}_SC{}_{}_1".format(c['model'], c['n_workers'], str(100), ce['seq_count'], self.date) # Unique stamp
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

    #########################################
    # Underscore methods for internal control of training and testing
    def train(self, steps=None):
        """
        """
        if not self.reloaded:
            self._create_model_dir()
        self._check_env_status()
        try:
            evaluate_every = self.config['testing'].get('evaluate_every')
            if evaluate_every is None: evaluate_every = 1000000
            config = dict(
                total_timesteps=steps if steps is not None else evaluate_every,
                tb_log_name=self._unique[:-2],
                reset_num_timesteps=False,
                seed=self.config['main']['seed'])
        
            self.reloaded = True

            for _ in range(self.n_steps//evaluate_every):
                self.model = self.model.learn(**config)
                self.test_runner.evaluate(self.config['testing']['test_timeout'])
                self._save()
                self.current_checkpoint += 1
        except KeyboardInterrupt:
            print('\n\n\nStopped training...\n')
            self._save()

    def _test(self, deterministic=True, rand=False):
        """
        Perform a single test step and return the mean actions and the rendered image
        """
        if self.done:
            self.test_state = self.env.reset()
            self.done = False
        
        action, s = self.model.predict(self.test_state, deterministic=True)
        print(action, s)
        self.test_state, _, _, _ = self.model.env.step(action)
        
        img = self.test_state.squeeze()
        
        mean_actions = None

        return mean_actions, img

    def run(self, steps=100, delay=0.001, reinit=False, rand=False, render=True, deterministic=True):
        """
        Run a small test loop`
        """
        
        self.done = True
        self._check_env_status()
        self.model.set_env(self.test_env)
        for _ in range(steps):
            _, img = self._test(deterministic=deterministic, rand=random)
            if render:
                # img.astype('uint8')
                img = np.clip(img, 0, 255)
                plt.cla(); plt.imshow(img); plt.show(); plt.pause(delay)
        self.model.set_env(self.env)
    
    def inverse_fold(self, target, budget=20):
        """
        Method for using the model to generate a nucleotide sequence
        solution given a target dot-bracket sequence
        """
        self.model.set_env(self.test_env)
        seq = Sequence(target, 0, 0)
        data = Dataset(sequences=[seq])
        self.model.env.set_attr('dataset', data)
        n = self.model.env.get_attr('next_target')[0]
        n()
        self.env.set_attr('meta_learning', False)
        
        self.test_state = self.model.env.reset()
        # Model
        for _ in range(budget):
            self.done = [False]
            while not self.done[0]:
                action, _ = self.model.predict(self.test_state)
                self.test_state, _, self.done, _ = self.model.env.step(action)
                s = self.model.env.get_attr('prev_solution')[0]
                if self.done[0]:
                    s.summary(True)

    def random_sampling(self, target, budget=100):
        """
        Randomly sample actions
        """
        self.model.set_env(self.test_env)
        seq = Sequence(target, 0, 0)
        data = Dataset(sequences=[seq])
        self.model.env.set_attr('dataset', data)
        n = self.model.env.get_attr('next_target')[0]()
        self.env.set_attr('meta_learning', False)
        
        self.test_state = self.model.env.reset()
        # Model
        for _ in range(budget):
            self.done = [False]
            episode_buffer = []
            while not self.done[0]:
                action = [self.model.env.action_space.sample()]
                self.test_state, _, self.done, _ = self.model.env.step(action)
                s = self.model.env.get_attr('prev_solution')[0]
                if self.done[0]:
                    s.summary(True)

        
if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    env = 'MountainCarContinuous-v0'
    b = SBWrapper(env)
    b.create_model()
    b.run()










    # def evaluate_testset(self, dataset='rfam_learn_test', budget=100, permute=False, show=False, pause=0):
    #     """
    #     Run
    #     """
    #     # driver = webdriver.Chrome('/home/andrius/chromedriver')
    #     if show:
    #         driver = create_browser('double')
    #     self.model.set_env(self.test_env)
    #     n_seqs=29 if dataset=='rfam_taneda' else 100
    #     if dataset=='rfam_learn_train':
    #         n_seqs=5000
        
    #     d = Dataset(dataset=dataset, start=1, n_seqs=n_seqs, encoding_type=self.config['environment']['encoding_type'])
    #     self.model.env.set_attr('dataset', d)
    #     self.model.env.set_attr('randomize', False)
    #     self.model.env.set_attr('meta_learning', False)
    #     get_seq = self.model.env.get_attr('next_target')[0]
    #     self.model.env.set_attr('current_sequence', 0)
    #     self.model.env.set_attr('permute', permute)

    #     self.test_state = self.model.env.reset()
    #     solved = []
        
    #     for n, seq in enumerate(d.sequences):
    #         print(n, '\n')
    #         get_seq()
    #         target = self.model.env.get_attr('target')[0]
    #         if show:
    #             show_rna(target.seq, 'AUAUAU', driver, 0)
    #             time.sleep(pause)
    #         end = False
    #         ep = 0
    #         for b in range(budget):
    #             self.done = [False]
    #             while not self.done[0]:
    #                 action, _ = self.model.predict(self.test_state)
    #                 self.test_state, _, self.done, _ = self.model.env.step(action)
    #                 s = self.model.env.get_attr('prev_solution')[0]
                    
    #                 if self.done[0]:
    #                     if show and ep%1==0:
    #                         show_rna(s.folded_design.seq, s.string, driver, 1)
                            
    #                         time.sleep(pause)
    #                     if s.hd <= 0:
    #                         s.summary(True)
    #                         solved.append([n, s, b+1, budget])
    #                         print('Solved sequence: {} in {}/{} iterations...'.format(n, b+1, budget))
    #                         end = True
    #                         ep += 1
    #                         # if show:
    #                         #     show_rna(s.target.seq, s.string, driver, 0, 'display')
    #             if end: break
    #     print('Solved ', len(solved), '/', len(d.sequences))

    #     self.write_test_results(solved, d)
        
    #     return solved

    # def timed_evaluation(self, dataset='rfam_learn_test', limit=60, permute=False, show=False, pause=0):
    #     """
    #     Run
    #     """
    #     if show:
    #         driver = create_browser('double')
    #     self.model.set_env(self.test_env)
    #     n_seqs=29 if dataset=='rfam_taneda' else 100
    #     if dataset=='rfam_learn_train':
    #         n_seqs=5000
        
    #     d = Dataset(dataset=dataset, start=1, n_seqs=n_seqs, encoding_type=self.config['environment']['encoding_type'])
    #     self.model.env.set_attr('dataset', d)
    #     self.model.env.set_attr('randomize', False)
    #     self.model.env.set_attr('meta_learning', False)
    #     get_seq = self.model.env.get_attr('next_target')[0]
    #     self.model.env.set_attr('current_sequence', 0)
    #     self.model.env.set_attr('permute', permute)

    #     self.test_state = self.model.env.reset()
    #     solved = []
    #     t_total = 0
        
    #     attempts = np.zeros([n_seqs], dtype=np.uint8)
    #     min_hd = np.ones([n_seqs], dtype=np.uint8) * 500
    #     time_taken = np.zeros([n_seqs])
    #     try:
    #         while t_total <= limit:
    #             ep_start = time.time()
    #             get_seq()
    #             target = self.model.env.get_attr('target')[0]
    #             if show:
    #                 show_rna(target.seq, 'AUAUAU', driver, 0)
    #                 time.sleep(pause)
    #             ep = 0
                
    #             self.done = [False]
    #             while not self.done[0]:
    #                 action, _ = self.model.predict(self.test_state)
    #                 self.test_state, _, self.done, _ = self.model.env.step(action)
    #                 s = self.model.env.get_attr('prev_solution')[0]
    #                 num = s.target.file_nr - 1
    #                 if self.done[0]:
    #                     if s.hd < min_hd[num]: min_hd[num] = s.hd
    #                     attempts[num] += 1
    #                     if show and ep%1==0:
    #                         show_rna(s.folded_design.seq, s.string, driver, 1)
    #                         time.sleep(pause)
    #                     t_episode = time.time() - ep_start
    #                     if s.hd <= 0:
    #                         dataset = self.model.env.get_attr('dataset')[0]
    #                         dataset.sequences.remove(s.target)
    #                         s.summary(True)
    #                         ep += 1
    #                     t_total += t_episode
    #                     time_taken[num] += t_episode
    #                     if s.hd <= 0:
    #                         solved.append([num, s, attempts[num], min_hd[num], round(time_taken[num],2)])
    #                         print('({}/{}) Solved sequence: {} in {} iterations, {:.2f} seconds...\n'.format(len(solved), n_seqs, num, attempts[num], time_taken[num]))
    #     except KeyboardInterrupt:
    #         pass
    #     print('Solved {}/{}'.format(len(solved), n_seqs))
    #     print(min_hd)

    #     self.write_test_results(solved, d)
        
    #     return solved

    # def write_test_results(self, results, dataset):
    #     """
    #     Writes the results of the test in ../results/<dataset>/<date>_<solved>.log
    #     """
    #     date = datetime.datetime.now().strftime("%m-%d_%H-%M")
    #     directory = os.path.join(settings.RESULTS, dataset.dataset)
    #     if not os.path.isdir(directory): os.makedirs(directory)
    #     filename = os.path.join(directory, '{}_{}.log'.format(date, len(results)))
    #     budget = results[0][3]
    #     with open(filename, 'w') as f:

    #         msg  = 'Dataset: {}, date: {}, solved {}/{} sequences with {} eval budget.\n'.format(
    #                 dataset.dataset, date, len(results), dataset.n_seqs, budget)
    #         # msg += 100 * 
    #         msg += ''.join(['=']*100) + '\n'
    #         f.write(msg)    
        
    #         for result in results:
    #             lines = result[1].summary()
    #             for line in lines:
    #                 f.write(line + '\n')
    #             f.write('Solved in: {}/{}\n'.format(result[2], budget))

    # def write_timed_results(self, solved, unsolved, dataset):
    #     """
    #     Writes the results of the test in ../results/<dataset>/<date>_<solved>.log
    #     """
    #     date = datetime.datetime.now().strftime("%m-%d_%H-%M")
    #     directory = os.path.join(settings.RESULTS, dataset.dataset)
    #     if not os.path.isdir(directory): os.makedirs(directory)
    #     filename = os.path.join(directory, '{}_{}.log'.format(date, len(results)))
    #     budget = results[0][3]
    #     with open(filename, 'w') as f:

    #         msg  = 'Dataset: {}, date: {}, solved {}/{} sequences with {} time budget.\n'.format(
    #                 dataset.dataset, date, len(results), dataset.n_seqs, budget)
    #         # msg += 100 * 
    #         msg += ''.join(['=']*100) + '\n'
    #         f.write(msg)    
        
    #         for result in solved:
    #             lines = result[1].summary()
    #             for line in lines:
    #                 f.write(line + '\n')
    #             f.write('Solved in: {}/{}\n'.format(result[2], budget))

     