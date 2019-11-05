import os, time, shutil, sys, re
try:
    sys.path.remove('/usr/local/lib/python3.6/site-packages')
except:
    pass

sys.path.append('/home/andrius/thesis/code/comparison/learna/src')
# from learna import Learna_fold
import traceback, random
import numpy as np

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph as pg
# from rlfold.definitions import learna_fold, Learna_fold
from rlfold.definitions import Dataset, Sequence, Solution, load_sequence
from rlfold.baselines import SBWrapper, get_parameters
from rlfold.definitions import colorize_nucleotides, highlight_mismatches, colorize_motifs
import rlfold.environments
from rlfold.utils import show_rna
import rlfold.settings as settings
# from rlfold.mcts import fold as mcts_fold
import imgkit
from PIL import Image
from .config import ConfigManager as config

import RNA, forgi
# RNA.cvar.uniq_ML = 1
param_files = [
    'rna_turner2004.par',
    'rna_turner1999.par',
    'rna_andronescu2007.par',
    'rna_langdon2018.par']

def set_vienna_param(param, value):
    setattr(RNA.cvar, param, value)

def set_vienna_params(n):

    params = os.path.join(settings.MAIN_DIR, 'utils', param_files[n])
    RNA.read_parameter_file(params)

"""
TODO:

    1. Make another fold function that does it a single time
    2. Config like in nnmidi
    3. Tables
    4. Solution display
    5. Get rlfold
"""



def get_icon(name):
    icon_path = os.path.join(settings.ICONS, name+'.svg')
    return icon_path

class ToggleButton(QPushButton):
    def __init__(self, parent, names, trigger, status=None, own_trigger=False):
        super(ToggleButton, self).__init__(parent=parent)
        self.par = parent
        self.setCheckable(True)
        self.names = names
        self.status = status
        self.setText('   '+ self.names[0])
        # self.status_change(False)
        if own_trigger:
            self.clicked[bool].connect(getattr(self.par, trigger))
        else:
            self.clicked[bool].connect(getattr(self.par, trigger))
        self.clicked[bool].connect(self.status_change)

        modes = [QIcon.Mode.Normal, QIcon.Mode.Normal, QIcon.Mode.Disabled]
        fns = [QIcon.State.Off, QIcon.State.On, QIcon.State.Off]
        icon = QIcon() # parent=self
        for i,name in enumerate(self.names):
            path = os.path.join(settings.ICONS, name+'.svg')
            icon.addPixmap(QPixmap(path), modes[i], fns[i]) #, fns[i]
        self.setIcon(icon)

    def status_change(self, toggled):
        tip = self.names[1] if toggled else self.names[0]
        self.setText('   '+tip)

    def stop(self):
        self.setChecked(False)
        self.status_change(False)


class ClickButton(QPushButton):
    def __init__(self, parent, name, triggers, status=None):
        super(ClickButton, self).__init__(parent=parent)
        self.setText('   '+ name)
        self.par = parent
        self.name = name
        self.setStatusTip(status)

        for trigger in triggers:
            self.clicked.connect(trigger)

        icon = QIcon() # parent=self
        path = os.path.join(settings.ICONS, name+'.svg')
        icon.addPixmap(QPixmap(path)) #, fns[i]
        self.setIcon(icon)


class ParameterSpinBox(QWidget):
    def __init__(self, parent, parameter):
        super(ParameterSpinBox, self).__init__(parent=parent)

        self.par = parent
        self.parameter = parameter
        self.translated = config.translate(parameter)
        self.scale = config.ranges[parameter]
        val = config.get(parameter)
        
        self.spin_box = QSpinBox()
        self.spin_box.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.slider = QSlider(Qt.Horizontal)
        self.set_ranges()
        self.spin_box.setValue(val)
        self.spin_box.setMinimumWidth(60)
        self.slider.setValue(self.find_nearest(val))
        self.slider.valueChanged[int].connect(self.value_changed)
        self.spin_box.valueChanged[int].connect(self.update_slider)
        
        name = ' ' * (20 - len(self.translated)) + self.translated + ':'
        self.label = QLabel(name)
        self.lay = QHBoxLayout()
        self.lay.addWidget(self.label)
        self.lay.addWidget(self.spin_box)
        self.lay.addWidget(self.slider)
        self.setLayout(self.lay)

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
    

class ParameterGroup(QGroupBox):
    def __init__(self, name, parent, parameters, additional=[]):
        super(ParameterGroup, self).__init__(name, parent=parent)
        self.parameter_dials = []
        self.par = parent

        self.lay = QVBoxLayout()
        for parameter in parameters:
            widget = ParameterSpinBox(parent, parameter)
            self.parameter_dials.append(widget)
            self.lay.addWidget(widget)
            self.lay.addSpacing(-25)
        for add in additional:
            self.lay.addWidget(add)
            self.lay.addSpacing(-25)
        self.lay.addSpacing(25)
        self.setLayout(self.lay)

class ParameterComboBox(QWidget):
    def __init__(self, name, parent, items, fn):
        super(ParameterComboBox, self).__init__(parent=parent)
        name = ' ' * (20 - len(name)) + name + ':'
        self.label = QLabel(name)
        self.selection = QComboBox(parent)
        self.selection.addItems(items)
        self.selection.currentIndexChanged.connect(fn)
        self.lay = QHBoxLayout(self)
        self.lay.addWidget(self.label)
        self.lay.addWidget(self.selection)

class ParameterCheckBox(QWidget):
    def __init__(self, name, parent, fn):
        super(ParameterCheckBox, self).__init__(parent=parent)
        self.fn = fn
        self.name = name
        name = ' ' * (23 - len(config.translate(name))) + config.translate(name) + ':'
        self.label = QLabel(name)
        self.check = QCheckBox()
        self.check.clicked.connect(self.was_clicked)
        self.lay = QHBoxLayout(self)
        self.lay.addWidget(self.label)
        self.lay.addWidget(self.check)
    
    def was_clicked(self):
        setattr(config, self.name, int(not config.get(self.name)))
        self.fn(self.name)

class CheckboxContainer(QWidget):
    def __init__(self, names, parent, fn, grid):
        super(CheckboxContainer, self).__init__(parent=parent)
        layout = QGridLayout(self)
        for i in range(grid[0]):
            for j in range(grid[1]):
                ind = grid[0]*i+j
                if ind >= len(names):
                    break
                name = names[ind]
                w = ParameterCheckBox(name, parent, fn)
                layout.addWidget(w, i, j, 1, 1)

class ParameterContainer(QGroupBox):
    def __init__(self, name, parent):
        super(ParameterContainer, self).__init__(name, parent=parent)

        self.par = parent

        self.fold_params = ParameterComboBox('Vienna params', self, param_files, set_vienna_params)
        self.checks = CheckboxContainer(
            names=['noGU', 'no_closingGU', 'uniq_ML'],
            parent=self.par,
            fn=self.par.parameter_changed, 
            grid=[2, 2])
        
        extra = [self.fold_params, self.checks]
        vienna = ['temperature', 'dangles']
        self.vienna_parameters = ParameterGroup('Vienna Parameters', self.par, vienna, extra)

        rlfold = ['TIME', 'N_SOLUTIONS', 'ATTEMPTS', 'WORKERS'] + ['permutation_budget', 'permutation_radius', 'permutation_threshold']
        self.rlfold_parameters = ParameterGroup('RL Parameters', self.par, rlfold)
        self.lay = QVBoxLayout()
        self.lay.addWidget(self.vienna_parameters)
        self.lay.addWidget(self.rlfold_parameters)
        self.setLayout(self.lay)

        
class MainWidget(QWidget):
    def __init__(self, parent):
        super(MainWidget, self).__init__(parent=parent)
        self.par = parent
        self.load = 1
        self.timer = QTimer()
        self.target = ''
        self.iter = 0
        self.total_iter = 0
        self.start = 0
        self.sequences = []
        self.solutions = []
        self.failed = []
        self.reinit = False
        self.dataset = 'eterna'

        self.running = False
        self.threadpool = QThreadPool()

        self.create_layout()
        self.model = self.load_model(*settings.model_dict['0'])
        self.fold_thread = Worker(self.model.single_fold)
        self.fold_thread.signals.result.connect(self.process_output)

        self.vfold_thread = Worker(self.vff)
        self.vfold_thread.signals.result.connect(self.process_output)

        # self.lfolder = Learna_fold()
        # self.lfold_thread = Worker(self.lfolder.run)
        # self.lfold_thread.signals.result.connect(self.process_output)
    

    def clear_all(self):
        # for i in reversed(range(self.lay.count())): 
        #     self.lay.itemAt(i).widget().setParent(None)
        self.total_iter = 0
        self.start = 0
        self.solutions = []
        self.failed = []

    def create_layout(self, redo=False):

        # Sequences
        if self.reinit:
            self.clear_all()
        else:
            self.lay = QGridLayout(self)

        self.sequence_input = QLineEdit(self)
        self.sequence_input.setMinimumWidth(1000)
        self.sequence_input.setPlaceholderText('Enter sequence in dot-bracket format.')
        self.sequence_input.textChanged.connect(self.parse_input)

        # Progress
        prlayout = QVBoxLayout()
        self.progress_view = QGroupBox('Fold')
        self.progress_bar = QProgressBar()
        self.progress_display = QTextEdit(self)
        self.progress_display.setText('Progress:')
        self.progress_display.setEnabled(False)
        prlayout.addWidget(self.progress_display)
        prlayout.addWidget(self.progress_bar)
        self.progress_view.setLayout(prlayout)
        
        # Display
        self.img_disp = pg.ImageView()
        self.img_disp.getView().setBackgroundColor('w')
        self.img_disp.ui.roiPlot.hide()
        self.img_disp.ui.roiBtn.hide()
        self.img_disp.ui.menuBtn.hide()
        self.img_disp.ui.histogram.hide()

        self.target_statistics = QTextEdit(self)
        self.target_statistics.setText('Target Statistics:')
        self.target_statistics.setEnabled(False)
                
        # Valid solution tab
        self.results_table = SolutionTable('None', self, None)
        self.solution_viz = SolutionWidget('Plots', self, None)
        self.valid_view = QGroupBox('Valid Solutions')
        vallay = QVBoxLayout()
        vallay.addWidget(self.solution_viz)
        vallay.addWidget(self.results_table)
        self.results_table.setMinimumHeight(150)
        self.valid_view.setLayout(vallay)

        # Failed solution tab
        self.failed_viz = FailedWidget('Failed Solutions', self, None)
        self.failed_table = FailedTable('None', self, None)
        self.failed_view = QGroupBox('Failed Solutions')
        self.failed_table.setMinimumHeight(150)
        faillay = QGridLayout()
        faillay.addWidget(self.failed_viz, 1, 1, 1, 1)
        faillay.addWidget(self.failed_table, 2, 1, 1, 1)
        self.failed_view.setLayout(faillay)

        # Tab container
        self.tabs = QTabWidget()
        self.tabs.addTab(self.valid_view, 't1')
        self.tabs.setTabText(0, 'Valid Solutions')
        self.tabs.addTab(self.failed_view, 't2')
        self.tabs.setTabText(1, 'Failed solutions')

        # Buttons
        self.fold_button  = ToggleButton(self, ['Fold', 'Stop'], 'loop')
        self.vfold_button = ToggleButton(self, ['VFold', 'Stop'], 'vloop')
        self.lfold_button = ToggleButton(self, ['boost', 'Stop'], 'boost')
        self.load_button  = ClickButton(self, 'Load', [self.load_seqs])
        self.save_button  = ClickButton(self, 'Save', [self.save_seqs])
        self.clear_button = ClickButton(self, 'Clear', [self.clear_all])
        self.buttons = QGroupBox('Control')

        self.model_selection = LabeledComboBox(self, 'Model', [x[0] for x in settings.model_dict.values()], 'Select model')
        self.model_selection.combo.currentIndexChanged.connect(self.reload_model)


        self.dataset_selection = DirectoryComboBox(
            parent=self, 
            directory=settings.DATA, 
            name='Dataset', 
            icon='Dataset', 
            triggers=[self.change_dataset])
        self.dataset_selection.selection.currentTextChanged.connect(self.change_dataset)

        self.data_selection = DirectoryComboBox(
            parent=self, 
            directory=os.path.join(settings.DATA, self.dataset), 
            name='Sample', 
            icon='Data', 
            triggers=[self.load_file])
        self.data_selection.selection.currentIndexChanged.connect(self.load_file)
        
        self.parameters = ParameterContainer('Parameters', self)
        
        btnl = QGridLayout()
        self.fold_button.setEnabled(False)
        self.vfold_button.setEnabled(False)
        btnl.addWidget(self.fold_button, 1, 1, 1, 1)
        btnl.addWidget(self.vfold_button, 1, 2, 1, 1)
        btnl.addWidget(self.load_button, 2, 1, 1, 1)
        btnl.addWidget(self.save_button, 2, 2, 1, 1)
        btnl.addWidget(self.clear_button, 3, 1, 1, 1)
        btnl.addWidget(self.lfold_button, 3, 2, 1, 1)
        btnl.addWidget(self.model_selection, 4, 1, 1, 2)
        btnl.addWidget(self.dataset_selection, 5, 1, 1, 2)
        btnl.addWidget(self.data_selection, 6, 1, 1, 2)
        self.buttons.setLayout(btnl)

        # Layout
        self.lay.addWidget(self.sequence_input, 1, 1, 1, 3)
        self.lay.addWidget(self.target_statistics, 2, 1, 1, 3)
        self.lay.addWidget(self.buttons, 3, 1, 1, 1)
        self.lay.addWidget(self.progress_view, 3, 2, 1, 2)
        self.lay.addWidget(self.img_disp, 1, 4, 3, 2)
        self.lay.addWidget(self.tabs, 4, 3, 3, 3)
        self.lay.addWidget(self.parameters, 4, 1, 3, 1)
        self.lay.setColumnMinimumWidth(3, 650)
        self.lay.setColumnMinimumWidth(1, 200)
        self.lay.setRowMinimumHeight(2, 100)
        self.lay.setColumnMinimumWidth(4, 400)

        self.reinit=True
        self.parameter_changed()
        self.sequence_input.setText(self.target)
        self.parse_input()
    
    def change_dataset(self, set):
        self.dataset = set
        self.data_selection.set_path(os.path.join(settings.DATA, self.dataset))

    # RLFOLD

    def boost(self, status):
        self.model.env.set_attr('boosting', status)
        self.load += 1
        self.model.model.env.set_attr('boosting', status)

    def loop(self, status):
        if status:
            self.model.prep(self.target, permute=True, verbose=False)
            self.start = time.time()
            self.timer.timeout.connect(self.fold)
            self.timer.start(1)
        else:
            self.reset()

    def fold(self):
        if not self.running:
            if (self.iter < config.ATTEMPTS or (time.time() - self.start) > config.TIME):
                self.running = True
                self.fold_thread.start()
                
                self.iter += 1
                self.total_iter += 1
            else:
                self.reset()
            self.update_progress()

    ## VIENNA
    def vloop(self, status):
        if status:
            self.start = time.time()
            self.timer.timeout.connect(self.vfold)
            self.timer.start(1)
        else:
            self.reset()

    def vff(self, target=None):
        t0 = time.time()
        seq = RNA.inverse_fold(None, target)
        return [seq[0], t0, 'v']
    
    def vfold(self):
        if not self.running:
            if self.iter < config.ATTEMPTS or (time.time() - self.start) > config.TIME:
                self.running = True
                self.vfold_thread.kwargs = {'target':self.target}
                self.vfold_thread.start()
                self.iter += 1
                self.total_iter += 1
            else:
                self.reset()
            self.update_progress()

    ## LEARNA
    def lloop(self, status):
        if status:
            self.lfolder.prep([self.target])
            self.start = time.time()
            self.timer.timeout.connect(self.lfold)
            self.timer.start(1)
        else:
            self.reset()

    def lfold(self):
        if not self.running:
            if self.iter < config.ATTEMPTS or (time.time() - self.start) > config.TIME:
                self.running = True
                # self.lfold_thread.kwargs = {'target':self.target}
                self.lfold_thread.start()
                self.iter += 1
                self.total_iter += 1
            else:
                self.reset()
            self.update_progress()

    def process_output(self, solution):
        self.running = False
        if type(solution[0]) is str:
            if len(solution) > 3:
                solution, hd, t0, t = solution
            else:
                solution, t0, t = solution
            native = 0 if t=='l' else 5
            source = 'learna' if t=='l' else 'RNAinverse'
            conf = self.model.model.env.get_attr('config')[0]
            target = Sequence(self.target)
            solution = Solution(
                target=target, 
                config=conf, 
                string=solution, 
                time=time.time()-t0, 
                source=source)
            if solution.hd == 0:
                self.new_solution(solution, native=native)
            else:
                self.new_failed(solution, native=native)
        if type(solution[0]) is str:
            solution, t0, t = solution
            native = 0 if t=='l' else 5
            source = 'learna' if t=='l' else 'RNAinverse'
            conf = self.model.model.env.get_attr('config')[0]
            target = Sequence(self.target)
            solution = Solution(
                target=target, 
                config=conf, 
                string=solution, 
                time=time.time()-t0, 
                source=source)
            if solution.hd == 0:
                self.new_solution(solution, native=native)
            else:
                self.new_failed(solution, native=native)
        else:
            for sol in solution:
                if sol.hd == 0:
                    self.new_solution(sol, native= self.load)
                else:
                    self.new_failed(sol, native=self.load)


    def load_image(self):
        show_rna(self.target, 'AUAUAU', None, 0, html='single')
        out = os.path.join(settings.RESULTS, 'images', '{}.jpg'.format(0))
        imgkit.from_file(os.path.join(settings.MAIN_DIR, 'display','single.html'), out)
        # time.sleep()
        img = np.array(Image.open(out))
        print(img.shape)
        self.img_disp.setImage(img[:, :int(img.shape[1]/7*3), :])

    def load_file(self, index):
        seq = load_sequence(num=int(index+1), dataset='eterna')
        self.sequence_input.setText(seq.seq)
        self.parse_input()
        
    def parse_input(self, str=None):

        dot_bracket_check = re.compile(r'[^.)()]').search # Regex for dotbrackets
        nucl_check = re.compile(r'[^AUGC]').search
        current_text = self.sequence_input.text().strip()
        if not bool(dot_bracket_check(current_text)) and len(current_text) > 0:
            self.target = current_text
            self.update_statistics()
            self.fold_button.setEnabled(True)
            self.vfold_button.setEnabled(True)
        else:
            self.target_statistics.setText('Invalid input.')

    def mfold(self):
        seq = mcts_fold(self.target)[0]
        conf = self.model.model.env.get_attr('config')[0]
        target = Sequence(self.target)
        solution = Solution(target, config=conf, string=seq[0])
        if solution.hd == 0:
            self.new_solution(solution, native=2)
        print(seq, solution.hd)

    def new_solution(self, solution, native=0):
        self.solutions.append(solution)
        self.results_table.new_solution(solution, native=native)
        self.solution_viz.new_solution(solution, native=native)

    def new_failed(self, solution, native=0):
        self.failed.append(solution)
        self.failed_table.new_solution(solution, native=native)
        self.failed_viz.new_solution(solution, native=native)


    def reset(self):
        self.running = False
        self.timer.stop()
        self.timer = QTimer()
        self.iter = 0
        self.vfold_button.stop()
        self.fold_button.stop()

    def load_model(self, directory, number, checkpoint=None):
        # n_envs = 6 if config.multi else 1
        trained_model = SBWrapper(
            'RnaDesign', directory).load_model(number, checkpoint=checkpoint, n_envs=config.WORKERS)
        self.progress_display.setText('Loaded model {}'.format(directory))
        
        return trained_model

    def reload_model(self, index):
        self.model = self.load_model(*settings.model_dict[str(index-1)])
        self.load += 1
        self.progress_display.setText('Loaded model {}'.format(directory))
        
    def update_statistics(self):
        target = Sequence(self.target)
        f, = forgi.load_rna(self.target)
        stems = len([s for s in f.stem_iterator()])
        stats = '{:15}: {}'.format('Target sequence', target.seq)
        stats += '\n{:15}: {}'.format('Motifs', target.markers)
        stats += '\n\n{:25}: {:6} | {:20}: {:3}'.format(
            'Target length', target.len, 'Hairpin loops', target.counter['H'])
        stats += '\n{:25}: {:.2f}% | {:20}: {:3}'.format(
            'Unpaired Nucleotides', target.db_ratio*100, 'Interior loops', target.counter['H'])
        self.target_statistics.setText(stats)
        self.load_image()

    def update_progress(self):
        self.progress_bar.setValue(self.iter)
        text = '{:20}: {:4}/{:4}'.format('Attempt', self.iter, config.ATTEMPTS)
        text += '\n{:20}: {:8}'.format('Total attempts', self.total_iter)
        text += '\n{:20}: {:8}'.format('Correct solutions', len(self.solutions))
        text += '\n{:20}: {:8}'.format('Failed solutions', len(self.failed))
        self.progress_display.setText(text)
    
    def load_seqs(self):
        sources = ['RNAinverse', 'MODENA', 'NUPACK', 'antaRNA', 'rnaMCTS', 'LEARNA', 'rlfold', 'mrlfold']
        name = QFileDialog.getOpenFileName()[0]
        config = self.model.model.env.get_attr('config')[0]
        with open(name, 'r') as seqfile:
            seqs = seqfile.readlines()[1:]
            target_string = seqs[0].strip().strip('\n')
            target = Sequence(target_string)
            self.target = target_string
            self.update_statistics()
            for seq in seqs[1:]:
                seq = seq.strip().strip('\n')
                seq, t, source = seq.split(' ')
                solution = Solution(target=target, config=config, string=seq, time=float(t), source=source)

                if solution.hd == 0:
                    self.new_solution(solution, native=sources.index(source))
                else:
                    self.new_failed(solution)
            self.load+= 1
            
    def save_seqs(self):
        save_dir = QFileDialog.getSaveFileName()[0]

        with open(save_dir, 'w') as seqfile:
            seqfile.write(self.target+'\n')
            for seq in self.sequences:
                seqfile.write(seq.string+'\n')

    def solution_selected(self, index):
        self.results_table.selectRow(index)

    def failed_selected(self, index):
        self.failed_table.selectRow(index)
    
    def row_selected(self, item):
        self.solution_viz.highlight(item.row())
    
    def parameter_changed(self, parameter=None):
        print(parameter)
        try:
            print(parameter, config.get(parameter))
        except:
            pass
        self.progress_bar.setRange(0, config.ATTEMPTS)
        if parameter in ['temperature', 'dangles', 'noGU', 'no_closingGU', 'uniq_ML']:
            set_vienna_param(parameter, config.get(parameter))
        if parameter in ['permutation_budget', 'permutation_radius', 'permutation_threshold']:
            print(parameter)
            conf = self.model.model.env.get_attr('config')[0]
            conf[parameter] = config.get(parameter)
            self.model.model.env.set_attr('config', conf)

class SolutionTable(QTableWidget):
    def __init__(self, name, parent, config):
        QTableWidget.__init__(self, 0, 6, parent=parent)
        self.setHorizontalHeaderLabels(['ID', 'Nucl seq', 'FE', 'Probability', 't', 'source'])
        self.setColumnWidth(1, 500)
        self.setColumnWidth(0, 50)
        self.itemClicked.connect(parent.row_selected)

    def new_solution(self, solution, native=0):
        count = self.rowCount()
        self.insertRow(count)
        if native != 0:
            self.setItem(count, 0, QTableWidgetItem('{:4}*{}'.format(count+1, native)))
        else:
            self.setItem(count, 0, QTableWidgetItem('{:6}'.format(count+1)))
        self.setItem(count, 1, QTableWidgetItem(solution.string))
        self.setItem(count, 2, QTableWidgetItem('{:.3f}'.format(solution.fe)))
        self.setItem(count, 3, QTableWidgetItem('{:.3f}'.format(solution.probability)))
        self.setItem(count, 4, QTableWidgetItem('{:.3f}'.format(solution.time)))
        self.setItem(count, 5, QTableWidgetItem('{:10}'.format(solution.source)))
        self.setRowHeight(count, 15)


    def solution_selected(self, ind):
        self.results_table.selectRow(ind)
        # Do something in the parent that will highlight the selected seq in the plots
    def failed_selected(self, ind):
        self.failed_table.selectRow(ind)
        # Do something in the parent that will highlight the selected seq in the plots

class FailedTable(QTableWidget):
    def __init__(self, name, parent, config):
        QTableWidget.__init__(self, 0, 6, parent=parent)
        self.setHorizontalHeaderLabels(['ID', 'Nucl seq', 'FE', 'HD', 'MD', 't', 'source'])
        self.setColumnWidth(1, 500)
        self.setColumnWidth(0, 50)
        self.itemClicked.connect(parent.failed_selected)

    def new_solution(self, solution, native=0):
        count = self.rowCount()
        self.insertRow(count)
        if native != 0:
            self.setItem(count, 0, QTableWidgetItem('{:4}*{}'.format(count+1, native)))
        else:
            self.setItem(count, 0, QTableWidgetItem('{:6}'.format(count+1)))
        self.setItem(count, 1, QTableWidgetItem(solution.string))
        self.setItem(count, 2, QTableWidgetItem('{:.3f}'.format(solution.fe)))
        self.setItem(count, 3, QTableWidgetItem('{:.3f}'.format(solution.hd)))
        self.setItem(count, 4, QTableWidgetItem('{:.3f}'.format(solution.md)))
        self.setItem(count, 5, QTableWidgetItem('{:.3f}'.format(solution.time)))
        self.setItem(count, 6, QTableWidgetItem('{:10}'.format(solution.source)))
        self.setRowHeight(count, 15)


    def solution_selected(self, ind):
        self.results_table.selectRow(ind)
        # Do something in the parent that will highlight the selected seq in the plots

    def failed_selected(self, ind):
        self.failed_table.selectRow(ind)


class Crosshair(pg.GraphicsObject):
    def __init__(self, parent, graph):
        pg.GraphicsObject.__init__(self)
        self.par = parent
        self.graph = graph
        self.xline = pg.InfiniteLine(angle=90, pen='k')
        self.yline = pg.InfiniteLine(angle=0, pen='k')
        self.par.addItem(self.xline)
        self.par.addItem(self.yline)

    def setPos(self, x, y):
        self.xline.setValue(x)
        self.yline.setValue(y)

class SolutionWidget(QGroupBox):
    def __init__(self, name, parent, config):
        QGroupBox.__init__(self, name, parent=parent)
        self.par = parent
        self.image_view = pg.ImageView()
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.getView().getViewBox().setBackground(None)
        self.image_view.getHistogramWidget().setBackground(None)
        self.solutions = []
        self.prob_data = []
        self.fe_data = []


        self.prob_view = pg.PlotWidget()
        self.prob_display = pg.ScatterPlotItem()
        self.prob_cr = Crosshair(self.prob_view, self.prob_display)
        self.prob_view.setXRange(-50, 5)
        self.prob_view.setYRange(0,1)
        self.prob_view.addItem(self.prob_display)
        self.prob_display.getViewBox().invertX(True)
        self.prob_display.sigClicked.connect(self.solution_selected)
        self.prob_view.setBackground(None)
        self.prob_view.getPlotItem().showGrid(True, True)
        self.prob_view.getPlotItem().setLabel(axis='bottom', text='Gibbs Free Energy')
        self.prob_view.getPlotItem().setLabel(axis='left', text='Probability of sequence')
        self.prob_view.getPlotItem().enableAutoScale()
        

        self.cnt_view = pg.PlotWidget()
        self.cnt_display = pg.ScatterPlotItem()
        self.cnt_cr = Crosshair(self.cnt_view, self.cnt_display)
        self.cnt_view.setXRange(-50, 5)
        self.cnt_view.setYRange(0,10)
        self.cnt_view.addItem(self.cnt_display)
        self.cnt_display.getViewBox().invertX(True)
        self.cnt_display.sigClicked.connect(self.solution_selected)
        self.cnt_view.setBackground(None)
        self.cnt_view.getPlotItem().showGrid(True, True)
        self.cnt_view.getPlotItem().setLabel(axis='bottom', text='Gibbs Free Energy', units='kCal/mol')
        self.cnt_view.getPlotItem().setLabel(axis='left', text='Distance to the centroid of the ensemble.')
        self.cnt_view.getPlotItem().enableAutoScale()

        self.ed_view = pg.PlotWidget()
        self.ed_display = pg.ScatterPlotItem()
        self.ed_cr = Crosshair(self.ed_view, self.ed_display)
        self.ed_view.setXRange(-50, 5)
        self.ed_view.setYRange(0,1)
        self.ed_view.addItem(self.ed_display)
        self.ed_display.getViewBox().invertX(True)
        self.ed_display.sigClicked.connect(self.solution_selected)
        self.ed_view.setBackground(None)
        self.ed_view.getPlotItem().showGrid(True, True)
        self.ed_view.getPlotItem().setLabel(axis='bottom', text='Gibbs Free Energy', units='kCal/mol')
        self.ed_view.getPlotItem().setLabel(axis='left', text='Ensemble defect.')
        self.ed_view.getPlotItem().enableAutoScale()

        self.gc_view = pg.PlotWidget()
        self.gcau(0, vals=[0,0,0,0])
        self.gc_view.setBackground(None)
        self.gc_view.setYRange(0,.8)
        self.gc_view.getPlotItem().showGrid(True, True)
        self.gc_view.getPlotItem().setLabel(axis='bottom', text='{:10}{:10}{:10}{:10}'.format('G', 'C', 'A','U'))
        self.gc_view.getPlotItem().setLabel(axis='left', text='Percentage')
        self.gc_view.setBackground(None)

        self.crs = [self.prob_cr, self.cnt_cr, self.ed_cr]
        self.lay = QHBoxLayout()
        self.lay.addWidget(self.prob_view)
        self.lay.addWidget(self.cnt_view)
        self.lay.addWidget(self.ed_view)
        self.lay.addWidget(self.gc_view)
        self.setLayout(self.lay)

    def new_solution(self, solution, native=0):
        native = native % 6
        c = {0:'g', 1:'b', 2:'r', 3:'y', 4:'c', 5:'m', 6:'k'}
        self.solutions.append(solution)
        color = pg.mkBrush(c[native])
        self.prob_display.addPoints([solution.fe], [solution.probability],brush=color)
        self.cnt_display.addPoints([solution.fe], [solution.centroid_dist], brush=color)
        self.ed_display.addPoints([solution.fe], [solution.ensemble_defect], brush=color)
        self.gcau(len(self.solutions)-1)

    def gcau(self, ind, vals=None):

        if vals is None:
            r = self.solutions[ind].gcau_content()
            vals = [r['G'],r['C'],r['A'],r['U']]
        try:
            self.gc_view.removeItem(self.g)   
            self.gc_view.removeItem(self.c)
            self.gc_view.removeItem(self.a) 
            self.gc_view.removeItem(self.u) 
        except:
            pass
        
        self.g = pg.BarGraphItem(x=[1], height=[vals[0]], width=.8, brush='g')
        self.c = pg.BarGraphItem(x=[2], height=[vals[1]], width=.8, brush='r')
        self.a = pg.BarGraphItem(x=[3], height=[vals[2]], width=.8, brush='b')
        self.u = pg.BarGraphItem(x=[4], height=[vals[3]], width=.8, brush='y')
        self.gc_view.addItem(self.g)
        self.gc_view.addItem(self.c)
        self.gc_view.addItem(self.a)
        self.gc_view.addItem(self.u)

    def update_image(self):
        pass

    def update_graphs(self):
        pass

    def highlight(self, index):
        for cr in self.crs:
            cr.setPos(cr.graph.data[index][0], cr.graph.data[index][1])

    def solution_selected(self, solution):
        try:
            ind = solution.ptsClicked[0]._index
            # self.prob_cr.setPos(x, y)
            self.par.failed_selected(ind)
            self.highlight(ind)
            r = self.solutions[ind].gcau_content()
            self.gcau(ind)

        except:
            print('fail')

class FailedWidget(QGroupBox):
    def __init__(self, name, parent, config):
        QGroupBox.__init__(self, name, parent=parent)
        self.par = parent
        self.image_view = pg.ImageView()
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.getView().getViewBox().setBackground(None)
        self.image_view.getHistogramWidget().setBackground(None)
        self.solutions = []
        self.mismatches = None

        self.md_c = {}
        self.mds = []
        self.hd_c = {}
        self.hds = []

        self.dist_view = pg.PlotWidget()
        self.dist_display = pg.ScatterPlotItem()
        self.dist_cr = Crosshair(self.dist_view, self.dist_display)
        self.dist_view.setXRange(-50, 5)
        self.dist_view.setYRange(0,1)
        self.dist_view.addItem(self.dist_display)
        self.dist_display.getViewBox().invertX(True)
        self.dist_display.sigClicked.connect(self.solution_selected)
        self.dist_view.setBackground(None)
        self.dist_view.getPlotItem().showGrid(True, True)
        self.dist_view.getPlotItem().setLabel(axis='bottom', text='Hamming distance')
        self.dist_view.getPlotItem().setLabel(axis='left', text='Mountain distance')
        self.dist_view.getPlotItem().enableAutoScale()

        # # self.heatmap = 
        # self.mm_view = pg.ImageView(view=pg.PlotItem())
        # self.mm_view.setPredefinedGradient('thermal')
        # self.mm_view.getView().showGrid(True, True)
        # self.mm_view.ui.roiPlot.hide()
        # self.mm_view.ui.roiBtn.hide()
        # self.mm_view.ui.menuBtn.hide()
        # self.mm_view.ui.histogram.hide()

        self.img_disp = pg.ImageView()
        self.img_disp.getView().setBackgroundColor('w')
        self.img_disp.ui.roiPlot.hide()
        self.img_disp.ui.roiBtn.hide()
        self.img_disp.ui.menuBtn.hide()
        self.img_disp.ui.histogram.hide()
        
        self.gc_view = pg.PlotWidget()
        self.gcau(0, vals=[0,0,0,0])
        self.gc_view.getPlotItem().showGrid(True, True)
        self.gc_view.setYRange(0,.8)
        self.gc_view.getPlotItem().setLabel(axis='bottom', text='{:10}{:10}{:10}{:10}'.format('G', 'C', 'A','U'))
        self.gc_view.getPlotItem().setLabel(axis='left', text='Percentage')
        self.gc_view.setBackground(None)

        self.crs = [self.dist_cr]
        self.lay = QGridLayout()
        self.lay.addWidget(self.dist_view, 1, 1, 1, 1)
        self.lay.addWidget(self.gc_view, 1, 2, 1, 1)
        self.lay.addWidget(self.img_disp, 1, 3, 1, 1)
        # self.lay.addWidget(self.mm_view, 2, 1, 1, 3)
        self.lay.setColumnMinimumWidth(3, 380)
        # self.lay.addWidget(self.ed_view)
        self.setLayout(self.lay)

    def new_solution(self, solution, native=0):
        native = native % 6
        # if self.mismatches is None:
        #     self.mismatches = np.zeros([2, solution.target.len])
        #     self.mismatches[0, :] = solution.target.to_binary()
        c = {0:'g', 1:'b', 2:'r', 3:'y', 4:'c', 5:'k', 6:'m'}
        self.solutions.append(solution)
        color = pg.mkBrush(c[native])
        if solution.hd not in self.hds:
            self.hds.append(solution.hd)
            self.hd_c[solution.hd] = 1
        else:
            self.hd_c[solution.hd] += 1

        # for mismatch in solution.mismatch_indices:
        #     self.mismatches[1, mismatch] += 0.01

        self.dist_display.addPoints([solution.hd], [solution.md],brush=color)
        # self.mm_view.setImage(1 - self.mismatches.T)
        # self.cnt_display.addPoints([solution.fe], [solution.centroid_dist], brush=color)
        # self.ed_display.addPoints([solution.fe], [solution.ensemble_defect], brush=color)
        self.gcau(len(self.solutions)-1)

    def gcau(self, ind, vals=None):

        if vals is None:
            r = self.solutions[ind].gcau_content()
            vals = [r['G'],r['C'],r['A'],r['U']]
        try:
            self.gc_view.removeItem(self.g)   
            self.gc_view.removeItem(self.c)
            self.gc_view.removeItem(self.a) 
            self.gc_view.removeItem(self.u) 
        except:
            pass
        
        self.g = pg.BarGraphItem(x=[1], height=[vals[0]], width=.8, brush='g')
        self.c = pg.BarGraphItem(x=[2], height=[vals[1]], width=.8, brush='r')
        self.a = pg.BarGraphItem(x=[3], height=[vals[2]], width=.8, brush='b')
        self.u = pg.BarGraphItem(x=[4], height=[vals[3]], width=.8, brush='y')
        self.gc_view.addItem(self.g)
        self.gc_view.addItem(self.c)
        self.gc_view.addItem(self.a)
        self.gc_view.addItem(self.u)

    def update_image(self):
        pass

    def update_graphs(self):
        pass

    def highlight(self, index):
        for cr in self.crs:
            cr.setPos(cr.graph.data[index][0], cr.graph.data[index][1])

    def solution_selected(self, solution):
        try:
            ind = solution.ptsClicked[0]._index
            # self.prob_cr.setPos(x, y)
            self.par.failed_selected(ind)
            self.highlight(ind)
            r = self.solutions[ind].gcau_content()
            self.gcau(ind)
            self.load_image(ind)

        except:
            print('fail')

    def load_image(self, index):
        solution = self.solutions[index].folded_design
        show_rna(solution, 'AUAUAU', None, 0, html='single')
        out = os.path.join(settings.RESULTS, 'images', '{}.jpg'.format(0))
        imgkit.from_file(os.path.join(settings.MAIN_DIR, 'display','single.html'), out)
        # time.sleep()
        img = np.array(Image.open(out))
        print(img.shape)
        self.img_disp.setImage(img[:, :int(img.shape[1]/7*3), :])

    def clear(self):
        # self.

class DirectoryComboBox(QWidget):
    def __init__(self, parent, directory, name, icon, triggers):
        super(DirectoryComboBox, self).__init__(parent=parent)
        self.par = parent
        self.icon = icon
        self.name = name
        self.fsm  = QFileSystemModel()
        self.selection = QComboBox()
        self.set_path(directory)

        self.label = QLabel(name)
        self.label.setFixedWidth(120)
        # self.label.setPixmap(QPixmap(get_icon(self.icon)))
        self.label.setText(name)
        
        ic = QLabel()
        ic.setPixmap(QPixmap(get_icon(self.icon)))
        ic.setFixedWidth(30)

        # self.new_button = ClickButton(
        #     parent=self.par, 
        #     name='New', 
        #     triggers=[getattr(self.par,triggers[0])],
        #     status='New')
        # self.new_button.setFixedWidth(30)

        # self.del_button = ClickButton(
        #     parent=self.par, 
        #     name='Delete', 
        #     triggers=[getattr(self.par,triggers[1])],
        #     status='Delete')
        # self.del_button.setFixedWidth(30)

        self.lay = QHBoxLayout()
        self.lay.addWidget(ic)
        self.lay.addWidget(self.label)
        self.lay.addWidget(self.selection)
        # self.lay.addWidget(self.new_button)
        # self.lay.addWidget(self.del_button)
        self.setLayout(self.lay)

    def set_path(self, directory):
        index = self.fsm.setRootPath(directory)
        self.selection.setModel(self.fsm)
        self.selection.setRootModelIndex(index)
        icon = QIcon(QPixmap(get_icon(self.icon)))
        self.selection.insertItem(0, icon, self.name)
        self.selection.setCurrentIndex(0)

    def check_existing(self, value):
        return True if self.selection.findData(value) >= 0 else False


class LabeledComboBox(QWidget):
    def __init__(self, parent, label, items, selection='Default'):
        super(LabeledComboBox, self).__init__(parent=parent)
        self.label = QLabel(label)

        self.combo = QComboBox()
        self.combo.addItems(items)
        icon = QIcon(QPixmap(get_icon(selection)))
        self.combo.insertItem(0, icon, label)
        self.combo.model().item(0).setEnabled(False)
        self.combo.setCurrentIndex(0)

        self.lay = QHBoxLayout()
        self.lay.addWidget(self.label)
        self.lay.addWidget(self.combo)
        self.setLayout(self.lay)

class LabeledCheckBox(QWidget):
    def __init__(self, parent, label, items, selection='Default'):
        super(LabeledComboBox, self).__init__(parent=parent)


class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data
    
    error
        `tuple` (exctype, value, traceback.format_exc() )
    
    result
        `object` data returned from processing, anything

    progress
        `int` indicating % progress 

    '''
    # finished = Signal()
    # error = Signal(tuple)
    # result = Signal(object)
    # progress = Signal(int)
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)

class Worker(QThread):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and 
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()   
        # self.rna = import RNA 

        # Add the callback to our kwargs
    #     # self.kwargs['progress_callback'] = self.signals.progress        
    # def __del__(self):
    #     self.wait()
    
    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        
        # Retrieve args/kwargs here; and fire processing using them
        try:
            # t0 = time.time()
            result = self.fn(*self.args, **self.kwargs)
            # print('thread')
            # seq = RNA.inverse_fold(None, **self.kwargs)
            # print('thread_done')
            # result = [seq[0], t0]
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


class VQWorker(QThread):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and 
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''
    # import RNA
    def __init__(self, fn, *args, **kwargs):
        super(VQWorker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals() 
        self.target = None
        # self.rna = import RNA 

        # Add the callback to our kwargs
        # self.kwargs['progress_callback'] = self.signals.progress        
    # def __del__(self):
    #     self.wait()
    # # @pySlot()

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        
        # Retrieve args/kwargs here; and fire processing using them
        try:
            t0 = time.time()
            print('thread')
            seq = RNA.inverse_fold(None, self.target)
            print('thread_done')
            # return [seq[0], t0]
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit([seq[0],t0])  # Return the result of the processing
