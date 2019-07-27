from PyQt5 import QtGui, QtCore, QtWidgets
import sys
import rusher.settings as settings
from rusher.interface import EnvDisplay, Graphs, Tables, BaselinesControl, Leaderboard
import gym
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack
# from rusher.environments import RusherStack, RusherMiniDiscrete
from rusher.baselines import BaselinesWrapper
from rusher.baselines.BaselinesWrapper import get_env_type
import stable_baselines
import webbrowser, os, yaml


class BaselinesGUI(BaselinesControl):
    """
    PyQt5 based interface for controlling forked and modified stable-baselines library. 
    
    Double inheritance:
        1. Control class that contains the button and console interface
        2. BaselinesWrapper that controls all the environment, model and policy objects, saving, loading etc.
    """ 
    def __init__(self):
        self.app = QtGui.QApplication(sys.argv)
        self.app.setApplicationName('Interactive Reinforcement Learning')
        super(BaselinesGUI, self).__init__()
           
        # Learning parameters
        self.use_display = True
        self.parameters = settings.parameters
        
        # Graph initialization

        # self.showMaximized()
        self.setup()
        self.env_name = self._env_selection.currentText()
        self.env_dir = None
        self.directory_search()
        # self.

    def setup(self):
        """
        Add the necessary modules and create the layout of the application
        """
        self.layout = QtWidgets.QGridLayout()

        # Widget groups
        self.env_display = EnvDisplay(parent=self)
        self.tables      = Tables(parent=self)
        self.leaderboard_ph = QtWidgets.QTreeWidget()
        # self.graphs      = Graphs(parent=self)
        # self.control = Control(parent=self)
        self.layout.setRowMinimumHeight(1, 500)
        if self.use_display:
            self.layout.setColumnMinimumWidth(2, 1000)
            self.layout.addWidget(self.env_display, 1, 2, 1, 1)
            self.setGeometry(0,0,1800,900)  
        else:
            self.setGeometry(0,0,800, 600)
        # self.layout.addWidget(self.graphs, 2, 1, 1, 1)
        self.layout.addWidget(self.widget, 1, 1, 1, 1) # Control Widget
        self.layout.addWidget(self.tables, 1, 3, 1, 1)
        
        self.layout.addWidget(self.leaderboard_ph, 2, 1, 1, 3)
        self.setLayout(self.layout)
    
    def interpret_message(self, message, **kwargs):
        """
        Interprets a message sent from a child widget to execute some function(s) in another child widget(s)
        Avoids cross-dependencies among children widgets
        """

        function_dictionary = dict(
            environment_changed=self.tables.reset_tables,
            set_status=self.set_status,
            initialize_env=self.initialize_env,
            leaderboard_changed=self.current_model_selected)

        # Call associated functions
        function_to_call = function_dictionary[message]
        if isinstance(function_to_call, list):
            for function in function_to_call:
                function()
        else:
            function_to_call(**kwargs)

    def _initialize_env(self, name, vectorized=False, image=False):
        """
        Initializes the current environment
        """
        print(f'Creating {name} environment...')
        self.wrapper = BaselinesWrapper(self.env_name, self.parameters)
        self.set_status(mode='initialized')
        # # Decorators
        # def bp_decorator(scale, discrete=False):
        #     """
        #     Decorator for my custom binpacking environments
        #     """
        #     def _init():
        #         if discrete:
        #             env = RusherMiniDiscrete
        #         else:
        #             env = RusherStack(self.parameters['Bin Packing']['Environment'], use_image_for_state=True, scale=scale)
        #         return env
        #     return _init

        # def vrep_decorator():
        #     """
        #     Decorator for my custom vrep environments
        #     Implement when needed
        #     """

        # def gym_env_decorator(name, rank, seed=0):
        #     """
        #     Decorator for gym environments
        #     """
        #     def _init():
        #         env = gym.make(name)
        #         env.seed(seed+rank)
        #         return env
        #     return _init

        # scale = 12.5
        # n_envs = 6
        # if vectorized:
        #     if env_type == 'bp':
        #         envs = [bp_decorator(scale) for _ in range(n_envs)]
        #     elif env_type == 'vrep':
        #         envs = [vrep_decorator() for _ in range(n_envs)]
        #     else:
        #         envs = [gym_env_decorator(name, 0) for _ in range(n_envs)]
            
        # vectorized_env = SubprocVecEnv(envs)

        # if image:
        #     vectorized_env = VecFrameStack(vectorized_env, n_stack=n_envs)

        # # Set status of other widgets
        
    def set_status(self, mode):
        """
        Enables/disables widgets based on what operation is being performed currently
        """
        self.tables.table_status(mode)
        self.button_status(mode)

    # Button controlled functions underscore

    def _pause(self):
        """
        Pauses the train_timer and stops the current episode
        """
        self.train_timer.stop()
        self.test_timer.stop()
        self.console.setText('Paused.')
        self.done = True
        self.button_status('paused')
        self.testState = None

        # Reset timers
        self.test_timer = QtCore.QTimer()
        self.train_timer = QtCore.QTimer()

    def _reset_env(self):
    
        """
        Reset env
        """
        self.train_timer.stop()
        self.test_timer.stop()
        self.done = True
        try:
            self.env.close()
            self.rewardWidget.plotItem().plot(clear=True)
            self.lossWidget.plotItem().plot(clear=True)
        except:
            pass
        self.button_status('uninitialized')
        self.terminate()

    def _train(self):
        """
        Start the training loop

        .train method should be redefined in the subclass, otherwise this will sample actions randomly
        """
        self.train_timer.timeout.connect(self.train)
        self.train_timer.start(1) # Timer in ms
        self.console.setText('Training...')
        self.button_status('running')
        # self.console.setPlainText(self.summary())

    def _test(self):
        """
        Run test loop one action at a time
        """
        self.test_timer.timeout.connect(self._test)
        self.test_timer.start(500)
        actions, images = self.wrapper._test()
        self.console.setText('Testing current policy...')
        self.button_status('running')
        
    def _stop(self):
        """
        Stop the training and reinitialize the loop
        """
        self.train_timer.stop()
        self.test_timer.stop()
        # self.initialize_env()

    def _save(self):
        """
        
        """
        
    def _load(self):
        """
        Loads parameters and files
        """
        selection = int(self._model_selection.currentText().split('_')[0])
        self.wrapper = BaselinesWrapper.load_num(self.env_name, selection)
        # self.wrapper = BaselinesWrapper(self.env_name, self.parameters, from_file=self.model_path)
    
    def parameter_summary(self, folder):
        pass

    
        
    def directory_search(self):
        """
        """
        env_type = get_env_type(self.env_name)
        self.env_dir = os.path.join(settings.TRAINED_MODELS, env_type, self.env_name)
        subfolders = os.listdir(self.env_dir)
        self._model_selection.clear()
        self._model_selection.addItems(subfolders)
        for folder in subfolders:
            curr = os.path.join(self.env_dir, folder)
            [print(x) for x in os.listdir(curr)] # Items 
    
    def _model_selection_changed(self):
        
        self.model_path = os.path.join(self.env_dir, self._model_selection.currentText())
        try:
            with open(os.path.join(self.model_path,'parameters.yml'), 'r') as f:
                self.parameters = yaml.load(f)
        except:
            print('Wrong selection')


    def _tensorboard(self):
        webbrowser.open('http://localhost:6006/')

    ### Application control
    def main(self):
        """
        Launch the app
        """
        self.show()
        self.raise_()
        sys.exit(self.app.exec_())

    def keyPressEvent(self, event):
        """
        Close the app with Esc
        """
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
            self.pause()
            try:
                self.env.close()
            except:
                pass

if __name__ == '__main__':
    disp = BaselinesGUI()
    disp.main()