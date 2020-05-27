from PySide2 import QtWidgets, QtGui, QtCore, QtSvg
import os, time, shutil, sys, re
from rlif.settings import ConfigManager as config
import numpy as np
import RNA

def set_vienna_params(n):

    params = os.path.join(config.MAIN_DIR, 'utils', 'parameters', config.param_files[n])
    RNA.read_parameter_file(params)

class ParameterSpinBox(QtWidgets.QWidget):
    def __init__(self, parent, parameter):
        super(ParameterSpinBox, self).__init__(parent=parent)

        self.par = parent
        self.parameter = parameter
        self.translated = config.translate(parameter)
        self.scale = config.ranges[parameter]
        val = config.get(parameter)
        self.spin_box = QtWidgets.QSpinBox()
        self.spin_box.setAlignment(QtGui.Qt.AlignRight | QtGui.Qt.AlignVCenter)
        self.slider = QtWidgets.QSlider(QtGui.Qt.Horizontal)
        self.set_ranges()
        self.spin_box.setValue(val)
        self.spin_box.setFixedWidth(60*config.scaling[0])
        self.slider.setValue(self.find_nearest(val))
        self.slider.valueChanged[int].connect(self.value_changed)
        self.spin_box.valueChanged[int].connect(self.update_slider)
        
        name = ' ' * (25 - len(self.translated)) + self.translated + ':'
        self.label = QtWidgets.QLabel(name)
        self.main_layout = QtWidgets.QHBoxLayout()
        self.main_layout.addWidget(self.label)
        self.main_layout.addWidget(self.spin_box)
        self.main_layout.addWidget(self.slider)
        self.setLayout(self.main_layout)

    def set_ranges(self):
        self.spin_box.setRange(self.scale[0], self.scale[-1])
        self.slider.setRange(0, len(self.scale)-1)
    
    def update_slider(self):
        value = self.find_nearest(self.spin_box.value())
        self.slider.setValue(value)

    def value_changed(self, value):
        value = self.scale[self.slider.value()]
        self.spin_box.setValue(value)
        self.par.parameter_changed(self.parameter)
        setattr(config, self.parameter, value)

    def find_nearest(self, value):
        array = np.asarray(self.scale)
        idx = (np.abs(array - value)).argmin()
        return idx
    
class ParameterGroup(QtWidgets.QGroupBox):
    def __init__(self, name, parent, parameters, additional=[]):
        super(ParameterGroup, self).__init__(name, parent=parent)
        self.parameter_dials = []
        self.par = parent
        self.main_layout = QtWidgets.QVBoxLayout()
        for parameter in parameters:
            widget = ParameterSpinBox(parent, parameter)
            self.parameter_dials.append(widget)
            self.main_layout.addWidget(widget)
            self.main_layout.addSpacing(-25)
        for add in additional:
            self.main_layout.addWidget(add)
            self.main_layout.addSpacing(-25)
        self.main_layout.addSpacing(25)
        self.setLayout(self.main_layout)

class ParameterComboBox(QtWidgets.QWidget):
    def __init__(self, name, parent, items, fn):
        super(ParameterComboBox, self).__init__(parent=parent)
        name = ' ' * (25 - len(name)) + name + ':'
        self.label = QtWidgets.QLabel(name)
        self.selection = QtWidgets.QComboBox(parent)
        self.selection.addItems(items)
        self.selection.currentIndexChanged.connect(fn)
        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.main_layout.addWidget(self.label)
        self.main_layout.addWidget(self.selection)

class ParameterCheckBox(QtWidgets.QWidget):
    def __init__(self, name, parent, fn):
        super(ParameterCheckBox, self).__init__(parent=parent)
        self.fn = fn
        self.name = name
        name = '{:25}'.format(config.translate(name) + ': ')
        self.label = QtWidgets.QLabel(name)
        self.check = QtWidgets.QCheckBox()
        self.check.clicked.connect(self.was_clicked)
        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.main_layout.addWidget(self.label)
        self.main_layout.addWidget(self.check)
    
    def was_clicked(self):
        setattr(config, self.name, int(not config.get(self.name)))
        self.fn(self.name)

class CheckboxContainer(QtWidgets.QWidget):
    def __init__(self, names, parent, fn, grid):
        super(CheckboxContainer, self).__init__(parent=parent)
        layout = QtWidgets.QGridLayout(self)
        for i in range(grid[0]):
            for j in range(grid[1]):
                ind = grid[0]*i+j
                if ind >= len(names):
                    break
                name = names[ind]
                w = ParameterCheckBox(name, parent, fn)
                layout.addWidget(w, i, j, 1, 1)

class ParameterContainer(QtWidgets.QGroupBox):
    def __init__(self, name, parent):
        super(ParameterContainer, self).__init__(name, parent=parent)

        self.par = parent
        self.fold_params = ParameterComboBox('Energy parameters', self, config.param_files, set_vienna_params)
        self.checks = CheckboxContainer(
            names=['noGU', 'no_closingGU'],
            parent=self.par,
            fn=self.par.parameter_changed, 
            grid=[2, 2])
        extra = [self.fold_params, self.checks]
        vienna = ['temperature', 'dangles']
        self.vienna_parameters = ParameterGroup('Vienna Parameters', self.par, vienna, extra)
        self.vienna_parameters.setFixedWidth(500)

        rlif = ['TIME', 'N_SOLUTIONS', 'ATTEMPTS', 'WORKERS'] + ['permutation_budget', 'permutation_radius', 'permutation_threshold']
        self.rlif_parameters = ParameterGroup('RL Parameters', self.par, rlif)
        self.main_layout = QtWidgets.QHBoxLayout()
        self.main_layout.addWidget(self.vienna_parameters)
        self.main_layout.addWidget(self.rlif_parameters)
        self.setLayout(self.main_layout)
