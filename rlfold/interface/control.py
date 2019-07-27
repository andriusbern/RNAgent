from PyQt5 import QtGui, QtCore, QtWidgets
import rusher.settings as settings

class BaselinesControl(QtWidgets.QWidget):
    """
    Contains all the elements needed for controlling the _train, _test procedure, environment loading and loading/saving models
    
    The subclass will need to implement the methods for all the behaviours, so that this is implementation is actually flexible
    """
    def __init__(self):
        super(BaselinesControl, self).__init__()
        # control_layout = QtWidgets.QGridLayout()
        self.train_timer = QtCore.QTimer()
        self.test_timer = QtCore.QTimer()
        self.done = False
        self.env_name = None

        # Environment selection
        self._env_label = QtWidgets.QLabel()
        self._env_label.setText('Environment:')
        self._env_selection = QtWidgets.QComboBox()
        self._env_selection.setCurrentText('Select the environment.')
        self._env_selection.addItems(['RusherStack', 'Pong-v0'])
        self._env_selection.currentTextChanged.connect(self._env_selection_changed)

        # Model selection
        self._model_label = QtWidgets.QLabel()
        self._model_label.setText('RL Model:')
        self._model_selection = QtWidgets.QComboBox()
        self._model_selection.addItems(['PPO2', 'A2C', 'DDPG'])
        self._model_selection.currentTextChanged.connect(self._model_selection_changed)

        # Policy selection
        self._policy_label = QtWidgets.QLabel()
        self._policy_label.setText('Policy type:')
        self._policy_selection = QtWidgets.QComboBox()
        self._policy_selection.addItems(['Fully connected', 'Convolutional', 'Recurrent'])
        self._policy_selection.currentTextChanged.connect(self._policy_selection_changed)

        # Buttons
        self._initialize_env_button = self._create_button(self._initialize_env, 'Create Environment')
        self._train_button = self._create_button(self._train, 'Train')
        self._test_button  = self._create_button(self._test, 'Test')
        self._pause_button = self._create_button(self._pause, 'Pause')
        self._stop_button  = self._create_button(self._stop, 'Stop')
        self._save_button  = self._create_button(self._save, 'Save')
        self._load_button  = self._create_button(self._load, 'Load')
        self._reset_button = self._create_button(self._reset_env, 'Reset')
        self._tensorboard_button = self._create_button(self._tensorboard, '_tensorboard')
        self.buttons = [self._initialize_env_button, self._train_button, self._test_button, self._pause_button,
                        self._save_button, self._load_button, self._reset_button, self._tensorboard_button]

        self.console = QtWidgets.QTextEdit()
        self.console.setText('Log.')

        self.widget = self.create_layout() # Returns a widget with a preset layout that can then be added to a larger application interface
        
    def create_layout(self):
        """
        Create a layout that can be added to another layout as a complete module (in the subclass)
        """
        control_widget = QtWidgets.QWidget()
        control_layout = QtWidgets.QGridLayout()
        control_layout.addWidget(self._env_label, 1, 1, 1, 1)
        control_layout.addWidget(self._env_selection, 1, 2, 1, 2)
        control_layout.addWidget(self._model_label, 2, 1, 1, 1)
        control_layout.addWidget(self._model_selection, 2, 2, 1, 2)
        control_layout.addWidget(self._policy_label, 3, 1, 1, 1)
        control_layout.addWidget(self._policy_selection, 3, 2, 1, 2)
        control_layout.addWidget(self._initialize_env_button, 4, 1, 1, 1)
        control_layout.addWidget(self._save_button, 4, 2, 1, 1)
        control_layout.addWidget(self._load_button, 4, 3, 1, 1)
        control_layout.addWidget(self._test_button, 5, 1, 1, 1)
        control_layout.addWidget(self._pause_button, 5, 2, 1, 1)
        control_layout.addWidget(self._stop_button, 5, 3, 1, 1)
        control_layout.addWidget(self._reset_button, 6, 1, 1, 2)
        control_layout.addWidget(self._tensorboard_button, 6, 1, 1, 3)
        control_layout.addWidget(self.console, 7, 1, 2, 3)
        control_widget.setLayout(control_layout)

        return control_widget

    @staticmethod
    def _create_button(function_to_connect, name):
        """
        Initialize a button with text and connect a member function to it
        """
        button = QtWidgets.QPushButton(name)
        button.setText(name)
        button.clicked.connect(function_to_connect)
        return button

    # def _env_selection_changed(self):
    #     """
    #     When the selection of environment changes, _load new parameters and send a message to the parent widget to update
    #     other child widgets
    #     """
    #     envName = self._env_selection.currentText()
    #     try:
    #         self.parameters = settings.parameters[envName]
    #     except:
    #         self.parameters = settings.parameters['Default']
    #     self.parent.interpret_message('environment_changed')
    #     self.parent.interpret_message('set_status', mode='initialized')

    def button_status(self, mode):
        """
        Sets the status of the buttons
        """
        button_status = {'uninitialized': [1, 0, 0, 0, 0, 1, 0],
                         'initialized':   [0, 1, 0, 0, 0, 1, 1],
                         'running'    :   [0, 0, 1, 0, 0, 0, 0],
                         'paused'     :   [0, 1, 0, 1, 1, 0, 1]}

        for button, state in zip(self.buttons, button_status[mode]):
            button.setEnabled(state)

    # To be overridden in the subclass
    def _train(self):
        pass

    def _test(self):
        # self.test_timer = 
        pass
    
    def _pause(self):
        pass
    
    def _stop(self):
        pass
    
    def _load(self):
        pass

    def _save(self):
        pass
    
    def _reset_env(self):
        pass

    def _initialze_env(self):
        pass
    
    def _tensorboard(self):
        pass

    def _env_selection_changed(self):
        """
        When the selection of environment changes, _load new parameters and send a message to the parent widget to update
        other child widgets
        """
        self.env_name = self._env_selection.currentText()
    
    def _model_selection_changed(self):
        pass

    def _policy_selection_changed(self):
        pass

    def _initialize_env(self):
        """
        Override
        """