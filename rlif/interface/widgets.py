from PySide2 import QtWidgets, QtGui, QtCore, QtSvg
import pyqtgraph as pg
import os, time, shutil, sys, re
import traceback, random
import numpy as np

import rlif.environments
from .parameters import ParameterContainer, CheckboxContainer
from rlif.rna import Dataset, DotBracket, Solution, load_sequence
from rlif.learning import Trainer, get_parameters
from rlif.rna import colorize_nucleotides, highlight_mismatches, colorize_motifs, load_fasta
from rlif.utils import draw, sol_draw
from rlif.learning import RLIF

from rlif.settings import ConfigManager as config

import RNA, forgi

def make_ruler(length, interval, start=0, add_markers=True):
    ruler  = ''.join(['  {:3}'.format(i) for i in range(start, length+1, interval)[1:]]) + '\n'
    if add_markers:
        ruler += ''.join(['    |' for i in range(start, length+1, interval)[1:]]) + '\n'
    return ruler

def set_vienna_param(param, value):
    setattr(RNA.cvar, param, value)

def set_vienna_params(n):

    params = os.path.join(config.MAIN_DIR, 'utils', 'parameters', config.param_files[n])
    RNA.read_parameter_file(params)

def get_icon(name):
    icon_path = os.path.join(config.ICONS, name+'.svg')
    return icon_path

def clicked(item, points):
    indexes = []
    for p in points:
        p = p.pos()
        lx = np.argwhere(item.data['x'] == p.x())
        ly = np.argwhere(item.data['y'] == p.y())
        i = np.intersect1d(lx, ly).tolist()
        indexes += i
    indexes = list(set(indexes))
    return indexes[0]

class WorkerSignals(QtCore.QObject):
    finished = QtCore.Signal()
    error = QtCore.Signal(tuple)
    result = QtCore.Signal(object)
    progress = QtCore.Signal(int)

class Worker(QtCore.QThread):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()   
    
    @QtCore.Slot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  
        finally:
            self.signals.finished.emit()  

class LabeledWidget(QtWidgets.QGroupBox):
    def __init__(self, widget, parent, name, main_layout='h'):
        super(LabeledWidget, self).__init__(name)
        main_layout = QtWidgets.QVBoxLayout if main_layout=='h' else QtWidgets.QHBoxLayout
        self.layout = main_layout(self)
        self.layout.addWidget(widget)
        self.setAlignment(QtGui.Qt.AlignCenter)

class MainWidget(QtWidgets.QWidget):
    def __init__(self, parent):
        super(MainWidget, self).__init__(parent=parent)
        self.par = parent
        
        self.init_vars()
        self.draw_mode = 0
        self.timer = QtCore.QTimer()
        self.dataset = 'eterna'
        self.create_layout()
        
        self.testing = False
        self.running = False

        self.threadpool = QtCore.QThreadPool()
        self.model = RLIF()
        self.rlif_thread = Worker(self.model.single_run)
        self.rlif_thread.signals.result.connect(self.process_output)
        self.step_thread = Worker(self.model.single_step)
        self.step_thread.signals.result.connect(self.process_step)

        self.vienna_rna_inverse_thread = Worker(self.vienna_fold_function)
        self.vienna_rna_inverse_thread.signals.result.connect(self.process_output)
        self.clear_all(True)

    def init_vars(self):
        self.load = 1
        self.iter = 0
        self.total_iter = 0
        self.start = 0
        self.sequences = []
        self.solutions = []
        self.failed = []
        self.targets = []
        self.target = None
        self.nucleotides = None
        self.solution = None
        self.current_target_index = 0

    def clear_all(self, init=False):
        if not init:
            reply = QtWidgets.QMessageBox.question(
                        self.par, 
                        'Clear all contents?', 
                        'Targets and results will be erased. Continue?', 
                        QtWidgets.QMessageBox.Yes, 
                        QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.No:
                return
        self.init_vars()
        self.enable_buttons(False)
        self.update_progress()
        self.solution_viz.clear()
        self.failed_viz.clear()
        self.results_table.reinit()
        self.failed_table.reinit()
        self.targets_table.reinit()
        self.img_display.load(os.path.join(config.UTILS, 'draw_rna', 'blank.svg'))

    def create_layout(self, redo=False):

        self.sequence_input = QtWidgets.QLineEdit(self)
        self.sequence_input.setMinimumWidth(5000*config.scaling[0])
        self.sequence_input.setPlaceholderText('Enter a nucleotide sequence (A/C/G/U) or a secondary RNA structure in dot-bracket format.')
        self.sequence_input.textChanged.connect(self.parse_input)
        self.sequence_input.returnPressed.connect(self.update_statistics)

        self.constraint_input = QtWidgets.QLineEdit(self)
        self.constraint_input.setMinimumWidth(5000*config.scaling[0])
        self.constraint_input.setPlaceholderText('Nucleotide sequence constraint')
        self.ruler = QtWidgets.QLineEdit(self)
        self.ruler.setMinimumWidth(5000*config.scaling[0])
        self.ruler.setPlaceholderText(make_ruler(500, 5, add_markers=False))
        self.ruler.setEnabled(False)
        # self.constraint_input.textChanged.connect(self.parse_input)

        scroll = QtWidgets.QScrollArea()
        self.inputs = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self.inputs)
        lay.addWidget(self.sequence_input)
        lay.addWidget(self.ruler)
        lay.addWidget(self.constraint_input)
        scroll.setWidget(self.inputs)

        # Progress
        self.progress_view = QtWidgets.QWidget()
        prlayout = QtWidgets.QVBoxLayout(self.progress_view)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_display = QtWidgets.QTextEdit(self)
        self.progress_display.setText('Progress:')
        self.progress_display.setEnabled(False)
        prlayout.addWidget(self.progress_display)
        prlayout.addWidget(self.progress_bar)

        # Target selection
        self.targets_table = TargetTable('Targets', self, None)
        self.targets_table.setFixedWidth(248*config.scaling[0])
        self.target_groupbox = LabeledWidget(self.targets_table, self, 'Target secondary RNA structures')
        
        # Sequence display
        self.img_tabs = QtWidgets.QTabWidget()
        self.img_display = QtSvg.QSvgWidget()
        self.cnt_disp = QtSvg.QSvgWidget()
        self.img_tabs.addTab(self.img_display, 't1')
        self.img_tabs.setTabText(0, 'MFE')
        self.img_tabs.addTab(self.cnt_disp, 't2')
        self.img_tabs.setTabText(1, 'Centroid')
        self.img_tabs.setFixedSize(400*config.scaling[0],400*config.scaling[1])
        self.img_tabs.currentChanged.connect(self.reload)
        self.img_display.setStatusTip('Minimum Free Energy (MFE) structure.')
        self.cnt_disp.setStatusTip('Centroid of the ensemble.')
        self.img_container = LabeledWidget(self.img_tabs, self, 'Secondary structure display')
        self.img_container.setFixedWidth(400)
        self.draw_mode_selection = LabeledComboBox(self, 'Mode', ['Nucleotides', 'Positional entropy'])
        self.draw_mode_selection.setFixedHeight(30)
        self.draw_mode_selection.combo.currentIndexChanged.connect(self.change_draw_mode)
        self.img_container.layout.addWidget(self.draw_mode_selection)

        self.target_statistics = QtWidgets.QTextEdit(self)
        self.target_statistics.setText('Target Statistics:')
        self.target_statistics_container = LabeledWidget(self.target_statistics, self, 'Secondary structure statistics')
        self.sequence_statistics = QtWidgets.QTextEdit(self)
        self.sequence_statistics_container = LabeledWidget(self.sequence_statistics, self, 'Nucleotide sequence')
                
        # Valid solution tab
        self.results_table = SolutionTable('None', self, None)
        self.solution_viz = SolutionWidget('Plots', self, None)
        self.valid_view = QtWidgets.QWidget()
        vallay = QtWidgets.QVBoxLayout(self.valid_view)
        vallay.addWidget(self.solution_viz)
        vallay.addWidget(self.results_table)
        self.results_table.setMinimumHeight(120*config.scaling[1])

        # Failed solution tab
        self.failed_viz = FailedWidget('Failed Solutions', self)
        self.failed_table = FailedTable('None', self, None)
        self.failed_view = QtWidgets.QWidget()
        faillay = QtWidgets.QGridLayout()
        faillay.addWidget(self.failed_viz, 1, 1, 1, 1)
        faillay.addWidget(self.failed_table, 2, 1, 1, 1)
        self.failed_view.setLayout(faillay)

        # Tab container
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.valid_view, 't1')
        self.tabs.setTabText(0, 'Valid Solutions')
        self.tabs.addTab(self.failed_view, 't2')
        self.tabs.setTabText(1, 'Failed solutions')

        # Buttons
        self.fold_button  = ToggleButton(self, ['Run', 'Stop'], 'loop', ['Run RLIF to generate nucleotide sequences for the target structure.', 'Stop.'])
        self.vienna_rna_inverse_button = ToggleButton(self, ['RNAinverse', 'Stop'], 'vienna_loop', ['Use RNAinverse.', 'Stop.'])
        self.test_button = ToggleButton(self, ['Test', 'Stop'], 'test', ['Run RLIF one nucleotide at a time.', 'Stop.'])
        self.load_button  = ClickButton(self, 'Load', [self.load_seqs], status='Load log files.')
        self.save_button  = ClickButton(self, 'Save', [self.save_seqs], status='Save solutions.')
        self.clear_button = ClickButton(self, 'Clear', [self.clear_all], status='Clear all solutions.')
        self.enter_button = ClickButton(self, 'New target', [self.update_statistics], status='Use the current RNA secondary structure as target.')
        self.random_button = ClickButton(self, 'Random', [self.random_nucleotide_seq], 'Generate a random nucleotide sequence.')
        self.buttons = QtWidgets.QGroupBox('Control')
        self.buttons.setFixedSize(250*config.scaling[0], 200)
        self.sequence_buttons = QtWidgets.QWidget()
        seqblay = QtWidgets.QHBoxLayout(self.sequence_buttons)
        seqblay.addWidget(self.enter_button)
        seqblay.addWidget(self.random_button)

        self.dataset_selection = DirectoryComboBox(
            parent=self, 
            directory=config.DATA,
            name='Dataset', 
            icon='Dataset', 
            triggers=[self.change_dataset])
        self.dataset_selection.selection.currentTextChanged.connect(self.change_dataset)

        self.load_dataset_button = ClickButton(self, 'Load Dataset', [self.load_dataset], 'Load dataset.')
        self.parameters = ParameterContainer('Parameters', self)
        self.container = QtWidgets.QTabWidget()
        self.container.addTab(self.progress_view, 't1')
        self.container.setTabText(0, 'Progress')
        self.container.addTab(self.parameters, 't2')
        self.container.setTabText(1, 'Parameters')

        btnl = QtWidgets.QGridLayout(self.buttons)
        btnl.addWidget(self.fold_button, 1, 1, 1, 1)
        btnl.addWidget(self.test_button, 1, 2, 1, 1)
        btnl.addWidget(self.vienna_rna_inverse_button, 2, 1, 1, 1)
        btnl.addWidget(self.clear_button, 2, 2, 1, 1)
        btnl.addWidget(self.save_button, 3, 1, 1, 1)
        btnl.addWidget(self.load_button, 3, 2, 1, 1)
        btnl.addWidget(self.dataset_selection, 4, 1, 1, 2)
        btnl.addWidget(self.load_dataset_button, 5, 1, 1, 2)

        self.stats = QtWidgets.QWidget()
        stl = QtWidgets.QHBoxLayout(self.stats)
        stl.addWidget(self.target_statistics_container)
        stl.addWidget(self.sequence_statistics_container)

        # Layout
        self.main_layout = QtWidgets.QGridLayout(self)
        self.main_layout.addWidget(self.buttons, 1, 1, 2, 1)
        self.main_layout.addWidget(self.stats, 2, 2, 1, 2)
        self.main_layout.addWidget(self.container, 3, 3, 1, 1)
        # self.main_layout.addWidget(self.sequence_input, 1, 2, 1, 4)
        self.main_layout.addWidget(scroll, 1, 2, 1, 4)
        self.main_layout.addWidget(self.sequence_buttons, 1, 6, 1, 1)
        self.main_layout.addWidget(self.img_container, 2, 6, 2, 1)
        self.main_layout.addWidget(self.tabs, 4, 2, 3, 5)
        self.main_layout.addWidget(self.target_groupbox, 3, 1, 4, 1)
        self.tabs.setFixedHeight(400)
        
        self.parameter_changed()
        self.parse_input()
        self.target_statistics.setText('Target sequence statistics.')
    

    ############
    # RLIF control
    def loop(self, status):
        if status:
            self.load += 1
            self.model.prep(self.target.seq)
            if self.testing:
                self.model.envs[0].reset()
                self.progress_bar.setRange(0, len(self.target.seq))
            self.start = time.time()
            self.timer.timeout.connect(self.run_rlif)
            self.timer.start(1)
        else:
            self.reset()

    def run_rlif(self):
        if self.testing:
            self.step_thread.start()
        else:
            if not self.running:
                if (self.iter < config.ATTEMPTS or (time.time() - self.start) > config.TIME):
                    self.running = True
                    self.rlif_thread.start()
                    self.iter += 1
                    self.total_iter += 1
                else:
                    self.reset()
                self.update_progress()

    def test(self, status):
        self.testing = status
        if status:
            self.model.prep(self.target.seq)
            self.model.envs[0].reset()
            self.progress_bar.setRange(0, len(self.target.seq))
            self.start = time.time()
            self.timer.timeout.connect(self.run_rlif)
            self.timer.start(1)
        else:
            self.reset()

    #########
    # VIENNA
    def vienna_loop(self, status):
        if status:
            self.start = time.time()
            self.timer.timeout.connect(self.vienna_rna_inverse)
            self.timer.start(1)
        else:
            self.reset()

    def vienna_fold_function(self, target=None):
        t0 = time.time()
        seq = RNA.inverse_fold(None, target)
        return [seq[0], t0, 'v']
    
    def vienna_rna_inverse(self):
        if not self.running:
            if self.iter < config.ATTEMPTS or (time.time() - self.start) > config.TIME:
                self.running = True
                self.vienna_rna_inverse_thread.kwargs = {'target':self.target.seq}
                self.vienna_rna_inverse_thread.start()
                self.iter += 1
                self.total_iter += 1
            else:
                self.reset()
            self.update_progress()

    def reset(self):
        self.running = False
        self.timer.stop()
        self.timer = QtCore.QTimer()
        self.iter = 0
        self.vienna_rna_inverse_button.stop()
        self.fold_button.stop()
        self.test_button.stop()
        self.testing = False

    ####################
    # Widget sync
    def new_solution(self, solution, color_id=0):
        self.sequences.append(solution)
        self.solutions.append(solution)
        self.solution = solution
        self.results_table.new_solution(solution, color_id=color_id)
        self.solution_viz.new_solution(solution, color_id=color_id)
        self.targets_table.update(self.current_target_index, solved=True)

    def new_failed(self, solution, color_id=0):
        self.sequences.append(solution)
        self.failed.append(solution)
        self.failed_table.new_solution(solution, color_id=color_id)
        self.failed_viz.new_solution(solution, color_id=color_id)
        self.targets_table.update(self.current_target_index, solved=False)

    def new_target(self, target, img=None):
        self.enable_buttons(True)
        self.targets.append(target)
        self.current_target_index = len(self.targets) - 1
        self.target = target
        self.targets_table.new_target(target)

    def target_selected(self, index):
        self.reset()
        self.current_target_index = index
        target = self.targets[index]
        self.target = target
        self.nucleotides = self.targets[index].nucleotides
        self.update_statistics(render=True, new_target=False)

    def solution_selected(self, index, row=True):
        solution = self.solutions[index]
        self.solution = solution
        if row:
            self.results_table.selectRow(index)
        self.nucleotides = solution.string
        self.solution_viz.highlight(index)
        self.update_statistics(new_target=False, solution=solution)
        self.load_image(solution.target.seq)

    def failed_selected(self, index, row=True):
        if row:
            self.failed_table.selectRow(index)
        self.failed_viz.highlight(index)
    
    def parameter_changed(self, parameter=None):
        self.progress_bar.setRange(0, config.ATTEMPTS)
        if parameter in ['temperature', 'dangles', 'noGU', 'no_closingGU']:
            set_vienna_param(parameter, config.get(parameter))
            self.nucl_fold(False)
        if parameter in ['permutation_budget', 'permutation_radius', 'permutation_threshold']:
            self.model.configure(**{parameter:config.get(parameter)})
    
    def enable_buttons(self, status):
        buttons = [self.vienna_rna_inverse_button, self.clear_button,
                   self.fold_button, self.save_button, self.test_button]
        [button.setEnabled(status) for button in buttons]

    #####################
    # Sequence parsing
    def random_nucleotide_seq(self):
        mapping = {0:'A', 1:'C', 2:'G', 3:'U'}
        length = random.randint(50, 350)
        sequence = ''.join([mapping[random.randint(0,3)] for x in range(length)])
        self.nucleotides = sequence
        self.nucl_fold(True, reset=True)

    def nucl_fold(self, btn=True, reset=False):
        """
        TODO: move to model
        """
        if self.nucleotides != None:
            target, _ = RNA.fold(self.nucleotides)
            conf = self.model.envs[0].config
            t = DotBracket(target)
            # t.nucleotides = self.nucleotides
            self.target = t
            self.solution = Solution(target=t, config=conf) #, string=self.nucleotides
            if reset:
                self.nucleotides = None
            self.update_statistics(True, new_target=btn, target=t)

    def process_output(self, solution):
        self.running = False
        if type(solution) is not str and type(solution) is not list:
            solution = solutions = [solution]

        if type(solution[0]) is str:
            if len(solution) > 3:
                solution, _, t0, t = solution
            else:
                solution, t0, t = solution
            color_id = 0 if t=='l' else 5
            source = 'learna' if t=='l' else 'RNAinverse'
            conf = self.model.envs[0].config
            solution = Solution(
                target=self.target, 
                config=conf, 
                string=solution, 
                time=time.time()-t0, 
                source=source)
            if solution.hd == 0:
                self.new_solution(solution, color_id=color_id)
                self.nucleotides = solution.string
                self.load_image()
            else:
                self.new_failed(solution, color_id=color_id)
            self.solution = solution
        else:
            for solution in solutions:
                if solution.hd == 0:
                    self.new_solution(solution, color_id= self.load)
                    self.solution = solution
                    self.nucleotides = solution.string
                    self.load_image()
                else:
                    self.new_failed(solution, color_id=self.load)
        self.update_progress()
        self.update_statistics(render=False, new_target=False)
        
    def process_step(self, solution):
        solution, done = solution
        print(solution.string)
        self.nucleotides = solution.string
        state = solution.get_state()
        msg = ''
        for i in range(state.shape[0]):
            binstr = ''.join([str(int(x)) for x in state[i, :, 0]])
            msg += binstr + '\n'
            self.progress_bar.setValue(solution.index+1)
        self.progress_display.setText(msg)
        self.sequence_statistics.setText(solution.string)
        self.load_image()
        if done:
            self.reset()
            self.process_output(solution)

    ##########
    ## Drawing
    def change_draw_mode(self, ind):
        self.draw_mode = ind
        self.load_image()

    def reload(self, num):
        self.load_image()

    def load_image(self, seq=None):
        if type(seq) is int:
            self.load_image(None)
        elif type(seq) is Solution:
            sequence = seq.string
        if self.img_display is self.img_tabs.currentWidget():
            sequence = self.target.seq if seq is None else seq
            disp = self.img_display
        else:
            if self.solution is not None:
                self.solution.compute_statistics()
                sequence = self.solution.centroid_structure
                disp = self.cnt_disp
        if self.draw_mode == 0:
            draw(sequence, self.nucleotides, col=None)
        elif self.draw_mode == 1:
            sol_draw(self.solution)
        img = os.path.join(config.UTILS, 'draw_rna', 'output.svg')
        disp.load(img)
        return img

    #############
    # Files

    def load_dataset(self):
        self.clear_all()
        self.to_load = [1, 100] if self.dataset != 'rfam_modena' else [1, 29]
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.load_file)
        self.timer.start(100)

    def change_dataset(self, data):
        self.dataset = data

    def load_file(self):
        ind = self.to_load[0]
        target = load_sequence(num=int(ind), dataset=self.dataset)
        self.target = target
        self.nucleotides = None
        self.to_load[0] += 1
        if ind == self.to_load[1]:
            self.timer.stop()
            self.timer = QtCore.QTimer()
        self.update_statistics(new_target=True, target=target)

    def load_seqs(self):
        sources1 = {'RNAinverse':'RNAinverse', 'MODENA':'MODENA', 'NUPACK':'NUPACK', 'antaRNA':'antaRNA', 'rnaMCTS':'rnaMCTS', 'LEARNA':'LEARNA', 'rlif':'rlif', 'rlif*':'rlif','rlfold':'rlif', 'brlfold':'rlif', 'mrlfold':'rlif'}
        sources = {'RNAinverse':0, 'MODENA':1, 'NUPACK':2, 'antaRNA':3, 'rnaMCTS':4, 'LEARNA':5, 'rlif':6}
        filename = QtWidgets.QFileDialog.getOpenFileName()[0]
        conf = self.model.envs[0].config
        ext = filename.split('.')[-1]
        if ext in ['fasta', 'fa']:
            sequences = load_fasta(filename, config=conf)
            self.to_load = sequences[::-1]
            self.progress_bar.setRange(0, len(sequences))
            self.progress_bar.setValue(0)
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.sequential_load)
            self.timer.start(1)

        else:
            with open(filename, 'r') as seqfile:
                seqs = seqfile.readlines()[1:]
                target_string = seqs[0].strip().strip('\n')
                target = DotBracket(target_string)
                self.target = target
                self.update_statistics(new_target=True)
                self.sequence_input.setText(target_string)
                for seq in seqs[1:]:
                    seq = seq.strip().strip('\n')
                    seq, t, source = seq.split(' ')
                    source = sources1[source]
                    # if seq != '---':
                    solution = Solution(target=target, config=conf, string=seq, time=float(t), source=source)

                    if solution.hd == 0:
                        self.new_solution(solution, color_id=sources[source])
                        self.nucleotides = solution.string
                        self.solution = solution
                        self.target = target
                        self.update_statistics(new_target=True)
                    else:
                        self.new_failed(solution)
        self.load+= 1
            
    def save_seqs(self):
        save_dir = QtWidgets.QFileDialog.getSaveFileName()[0]

        with open(save_dir, 'w') as seqfile:
            seqfile.write('{} valid solutions, {} invalid solutions\n'.format(len(self.solutions), len(self.failed)))
            seqfile.write(self.target.seq+'\n')
            for seq in self.sequences:
                seqfile.write(seq.string + ' {:.2f} '.format(seq.time) + seq.source +'\n')
    
    def sequential_load(self):
        """
        Workaround for non-threaded loading to keep the UI responsive
        """
        if len(self.to_load) < 1:
            self.timer.stop()
            self.timer = QtCore.QTimer()
        else:
            sequence = self.to_load.pop()
            self.nucleotides = sequence.string
            self.target = sequence.target
            self.new_target(sequence.target)
            self.new_solution(sequence, color_id=self.load)
            self.update_statistics(new_target=False)
            self.progress_bar.setValue(self.progress_bar.value()+1)

    ###############
    # IO
    def parse_input(self, str=None):
        dot_bracket_check = re.compile(r'[^.)()]').search # Regex for dotbrackets
        nucl_check = re.compile(r'[^AUGCaugc]').search
        current_text = self.sequence_input.text().strip()
        if not bool(dot_bracket_check(current_text)) and len(current_text) > 0:
            self.nucleotides = None
            target = DotBracket(current_text)
            self.update_statistics(new_target=False, target=target)
        elif not bool(nucl_check(current_text)) and len(current_text) > 0:
            self.nucleotides = current_text.upper()
            self.nucl_fold(False)
            self.constraint_input.setText('-'*len(current_text))
        else:
            self.target_statistics.setText('Invalid input.')
            self.constraint_input.setText('')

    def update_statistics(self, render=True, new_target=True, target=None, solution=None):
        if target is None:
            target = self.target
            nucleotides = self.nucleotides
        else:
            self.target = target
            nucleotides = target.nucleotides

        if render:
            self.load_image()

        if new_target:
            self.new_target(target)

        if solution is not None:
            target = solution.target
            self.target = solution.target
            self.nucleotides = nucleotides = solution.string
            
        # f, = forgi.load_rna(target.seq)
        
        ruler = make_ruler(80, 5)
        stats = ruler
        nstats = ruler

        for i in range(len(target.seq)//80+1):
            stats += target.seq[i*80:(i+1)*80] + '\n'
            if nucleotides is not None:
                nstats += nucleotides[i*80:(i+1)*80] + '\n'
        nstats += '\n'+ target.name

        stats += '\n{:25}: {:6} | {:20}: {:3}'.format(
            'Target length', target.len, 'Hairpin loops', target.counter['H'])
        stats += '\n{:25}: {:.2f}% | {:20}: {:3}\n\n'.format(
            'Unpaired Nucleotides', target.percent_unpaired*100, 'Interior loops', target.counter['H'])
        
        self.target_statistics.setText(stats)
        self.sequence_statistics.setText(nstats)

    def update_progress(self):
        self.progress_bar.setValue(self.iter)
        text = '{:20}: {:4}/{:4}'.format('Attempt', self.iter, config.ATTEMPTS)
        text += '\n{:20}: {:8}'.format('Total attempts', self.total_iter)
        text += '\n{:20}: {:8}'.format('Correct solutions', len(self.solutions))
        text += '\n{:20}: {:8}'.format('Failed solutions', len(self.failed))
        self.progress_display.setText(text)


class ToggleButton(QtWidgets.QPushButton):
    def __init__(self, parent, names, trigger, status=None, own_trigger=False):
        super(ToggleButton, self).__init__(parent=parent)
        self.par = parent
        self.setCheckable(True)
        self.names = names
        self.status = status
        self.setText('  '  + '{:10}'.format(self.names[0]))
        if own_trigger:
            self.clicked[bool].connect(getattr(self.par, trigger))
        else:
            self.clicked[bool].connect(getattr(self.par, trigger))
        self.clicked[bool].connect(self.status_change)
        modes = [QtGui.QIcon.Mode.Normal, QtGui.QIcon.Mode.Normal, QtGui.QIcon.Mode.Disabled]
        fns = [QtGui.QIcon.State.Off, QtGui.QIcon.State.On, QtGui.QIcon.State.Off]
        icon = QtGui.QIcon() # parent=self
        for i,name in enumerate(self.names):
            path = os.path.join(config.ICONS, name+'.svg')
            icon.addPixmap(QtGui.QPixmap(path), modes[i], fns[i]) #, fns[i]
        self.setIcon(icon)

    def status_change(self, toggled):
        tip = self.names[1] if toggled else self.names[0]
        self.setText('  ' + '{:10}'.format(tip))
        self.setStatusTip(self.status[toggled])

    def stop(self):
        self.setChecked(False)
        self.status_change(False)


class ClickButton(QtWidgets.QPushButton):
    def __init__(self, parent, name, triggers, status=None):
        super(ClickButton, self).__init__(parent=parent)
        self.setText('  '+ '{:10}'.format(name))
        self.par = parent
        self.name = name
        self.setStatusTip(status)

        for trigger in triggers:
            self.clicked.connect(trigger)

        icon = QtGui.QIcon() # parent=self
        path = os.path.join(config.ICONS, name+'.svg')
        icon.addPixmap(QtGui.QPixmap(path)) #, fns[i]
        self.setIcon(icon)


    def vienna_params(self, param):
        set_vienna_params(param)
        self.par.nucl_fold(False)
        self.par.update_statistics(new_target=False)
        
class SolutionTable(QtWidgets.QTableWidget):
    def __init__(self, name, parent, config):
        QtWidgets.QTableWidget.__init__(self, 0, 6, parent=parent)
        self.par = parent
        self.setColumnWidth(1, 500)
        self.setColumnWidth(0, 50)
        self.currentCellChanged.connect(self.row_selected)
    
    def new_solution(self, solution, color_id=0):
        count = self.rowCount()
        self.insertRow(count)
        self.setItem(count, 0, QtWidgets.QTableWidgetItem('{:6}'.format(count+1)))
        self.setItem(count, 1, QtWidgets.QTableWidgetItem(solution.string))
        self.setItem(count, 2, QtWidgets.QTableWidgetItem('{:.3f}'.format(solution.fe)))
        self.setItem(count, 3, QtWidgets.QTableWidgetItem('{:.3f}'.format(solution.probability)))
        self.setItem(count, 4, QtWidgets.QTableWidgetItem('{:.3f}'.format(solution.time)))
        self.setItem(count, 5, QtWidgets.QTableWidgetItem('{:10}'.format(solution.source)))
        self.setRowHeight(count, 15)

    def row_selected(self, ind):
        self.par.solution_selected(ind, row=False)

    def reinit(self):
        self.clear()
        self.clearContents()
        for row in reversed(range(self.rowCount())):
            self.removeRow(row)
        self.setHorizontalHeaderLabels(['ID', 'Nucleotide sequence', 'Free energy', 'Probability', 'Time taken', 'Source'])


class FailedTable(QtWidgets.QTableWidget):
    def __init__(self, name, parent, config):
        QtWidgets.QTableWidget.__init__(self, 0, 7, parent=parent)
        self.par = parent
        self.setColumnWidth(1, 500)
        self.setColumnWidth(0, 50)
        self.currentCellChanged.connect(self.row_selected)

    def new_solution(self, solution, color_id=0):
        count = self.rowCount()
        self.insertRow(count)
        if color_id != 0:
            self.setItem(count, 0, QtWidgets.QTableWidgetItem('{:4}*{}'.format(count+1, color_id)))
        else:
            self.setItem(count, 0, QtWidgets.QTableWidgetItem('{:6}'.format(count+1)))
        self.setItem(count, 1, QtWidgets.QTableWidgetItem(solution.string))
        self.setItem(count, 2, QtWidgets.QTableWidgetItem('{:.3f}'.format(solution.fe)))
        self.setItem(count, 3, QtWidgets.QTableWidgetItem('{:.3f}'.format(solution.hd)))
        self.setItem(count, 4, QtWidgets.QTableWidgetItem('{:.3f}'.format(solution.md)))
        self.setItem(count, 5, QtWidgets.QTableWidgetItem('{:.3f}'.format(solution.time)))
        self.setItem(count, 6, QtWidgets.QTableWidgetItem('{:10}'.format(solution.source)))
        self.setRowHeight(count, 15)

    def row_selected(self, ind):
        self.par.failed_selected(ind, row=False)

    def reinit(self):
        self.clear()
        self.clearContents()
        for row in reversed(range(self.rowCount())):
            self.removeRow(row)
        self.setHorizontalHeaderLabels(['ID', 'Nucleotide sequence', 'Free energy', 'HD', 'MD', 'Time Taken', 'Source'])

class TargetTable(QtWidgets.QTableWidget):
    def __init__(self, name, parent, config):
        QtWidgets.QTableWidget.__init__(self, 0, 5, parent=parent)
        self.par = parent
        self.setColumnWidth(0, 70)
        self.setColumnWidth(1, 30)
        self.setColumnWidth(2, 30)
        self.setColumnWidth(3, 30)
        self.setColumnWidth(4, 85)

        self.currentCellChanged.connect(self.row_selected)

    def new_target(self, target):
        size = 70 
        row = self.rowCount()
        self.insertRow(row)
        self.setRowHeight(row, size)
        self.setColumnWidth(0, size)

        thumb = QtSvg.QSvgWidget(self)
        thumb.setFixedSize(size, size)
        img = os.path.join(config.UTILS, 'draw_rna', 'output.svg')
        thumb.load(img)
        
        widget = QtWidgets.QTableWidgetItem
        self.setCellWidget(row, 0, thumb)
        self.setItem(row, 1, widget('{}'.format(target.len)))
        self.setItem(row, 2, widget(str(0)))
        self.setItem(row, 3, widget(str(0)))
        self.setItem(row, 4, widget(target.name))

    def row_selected(self, item):
        if type(item) is int:
            ind = item
            self.par.target_selected(ind)
        else:
            ind = item.row()
            self.par.target_selected(ind)

    def update(self, row, solved):
        index = 2 if solved else 3
        item = self.item(row, index)
        item.setText(str(int(item.text()) + 1))

    def reinit(self):
        self.clear()
        self.clearContents()
        for row in reversed(range(self.rowCount())):
            self.removeRow(row)
        self.setHorizontalHeaderLabels(['Seq', 'len', '%S', '%F', 'Source'])

class Crosshair(pg.GraphicsObject):
    def __init__(self, parent, graph):
        pg.GraphicsObject.__init__(self)
        self.par = parent
        self.graph = graph
        self.xline = pg.InfiniteLine(angle=90, pen='k')
        self.yline = pg.InfiniteLine(angle=0, pen='k')
        self.par.addItem(self.xline)
        self.par.addItem(self.yline)

    def set_pos(self, x, y):
        self.xline.setValue(x)
        self.yline.setValue(y)

class SolutionWidget(QtWidgets.QGroupBox):
    def __init__(self, name, parent, config):
        QtWidgets.QGroupBox.__init__(self, parent=parent)
        self.par = parent
        self.solutions = []
        # Plots
        self.nucl_content_plot = GCAUBarPlot(self)
        self.prob_plot = CrosshairPlot(
            parent=self, 
            click_function=self.par.solution_selected,
            x_label='Gibbs Free Energy',
            y_label='Probability of sequence')

        self.centroid_d_plot = CrosshairPlot(
            parent=self,
            click_function=self.par.solution_selected,
            x_label='Gibbs Free Energy',
            y_label='Ensemble Defect',
            y_range=(0,10))

        self.ed_plot = CrosshairPlot(
            parent=self,
            click_function=self.par.solution_selected,
            x_label='Gibbs Free Energy',
            y_label='Distance to the centroid of the ensemble.')
        
        self.plots = [self.prob_plot, self.centroid_d_plot, self.ed_plot]
        self.main_layout = QtWidgets.QHBoxLayout()
        self.main_layout.addWidget(self.prob_plot.view)
        self.main_layout.addWidget(self.centroid_d_plot.view)
        self.main_layout.addWidget(self.ed_plot.view)
        self.main_layout.addWidget(self.nucl_content_plot)
        self.setLayout(self.main_layout)
        self.highlight()

    def new_solution(self, solution, color_id=0):
        colors = {0:'g', 1:'b', 2:'r', 3:'y', 4:'colors', 5:'m', 6:'k'}
        color = pg.mkBrush(colors[color_id % 6])
        self.solutions.append(solution)
        self.prob_plot.addPoints([solution.fe], [solution.probability],brush=color)
        self.centroid_d_plot.addPoints([solution.fe], [solution.centroid_dist], brush=color)
        self.ed_plot.addPoints([solution.fe], [solution.ensemble_defect], brush=color)
        self.highlight(len(self.solutions)-1, cross=False)

    def clear(self):
        [plot.clear() for plot in self.plots]
        self.gcau()
        self.solutions = []
        for plot in self.plots:
            plot.crosshair.set_pos(0, 0)

    def gcau(self, ind=None):
        if ind is None:
            heights = [0, 0, 0, 0]
        else:
            percentage = self.solutions[ind].gcau_content()
            heights = [percentage['G'], percentage['C'], percentage['A'], percentage['U']]
        self.nucl_content_plot.bar_plot(heights=heights, colors=['r','g','y','b'])

    def highlight(self, index=None, cross=True):
        if index is not None and cross:
            [plot.point_selected(index) for plot in self.plots]
        self.gcau(index)


class GCAUBarPlot(pg.PlotWidget):
    """
    Custom bar plot for AUGC content
    """
    def __init__(self, parent, width=150):
        pg.PlotWidget.__init__(self)
        self.bars = []
        self.getPlotItem().getViewWidget().setFixedWidth(150)
        self.getPlotItem().getAxis('bottom').setStyle(showValues=False)
        # self.bar_plot(heights=[0,0,0,0], colors=['r','g','y','b'])
        self.setBackground(None)
        self.setYRange(0,.8)
        self.getPlotItem().showGrid(True, True)
        self.getPlotItem().setLabel(axis='bottom', text='{:10}{:10}{:10}{:10}'.format('G', 'C', 'A','U'))
        self.getPlotItem().setLabel(axis='left', text='Nucleotide content')

    def bar_plot(self, heights, colors, width=.8, clear=True):
        if clear:
            for item in self.bars:
                self.removeItem(item)
                del item

        for i, height in enumerate(heights):
            bar = pg.BarGraphItem(x=[i+1], height=[height], width=width, brush=colors[i])
            self.bars.append(bar)
            self.addItem(bar)

class CrosshairPlot(pg.ScatterPlotItem):
    def __init__(self, parent, click_function, x_label='', y_label='', 
                 x_range=(-50,5), y_range=(0,1), invert=True, autoscale=True):
        pg.ScatterPlotItem.__init__(self)

        self.par = parent
        self.click_function = click_function
        self.view = pg.PlotWidget()
        self.view.getViewBox().invertX(invert)
        self.view.setXRange(*x_range)
        self.view.setYRange(*y_range)
        self.view.addItem(self)
        self.view.setBackground(None)
        self.view.getPlotItem().showGrid(True, True)
        self.view.getPlotItem().setLabel(axis='bottom', text=x_label)
        self.view.getPlotItem().setLabel(axis='left', text=y_label)
        self.sigClicked.connect(self.point_clicked)
        self.crosshair = Crosshair(self.view, self)
        if autoscale:
            self.view.getPlotItem().enableAutoScale()

    def point_clicked(self, marks, points):
        index = clicked(marks, points)
        self.click_function(index)
    
    def point_selected(self, index):
        print(self.data[index])
        self.crosshair.set_pos(x=self.data[index][0], y=self.data[index][1])
        

class FailedWidget(QtWidgets.QGroupBox):
    def __init__(self, name, parent):
        QtWidgets.QGroupBox.__init__(self, parent=parent)
        self.par = parent
        self.solutions = []
        
        self.distance_plot = CrosshairPlot(
            parent=self,
            click_function=self.par.failed_selected,
            x_label='Hamming Distance',
            y_label='Mountain Distance',
            x_range=(0, 20),
            y_range=(0, 20),
            invert=False,
            autoscale=False)

        self.nucl_content_plot = GCAUBarPlot(self)
        self.img_display = QtSvg.QSvgWidget()
        self.img_display.setFixedSize(200*config.scaling[0], 200*config.scaling[1])
        
        self.main_layout = QtWidgets.QGridLayout()
        self.main_layout.addWidget(self.distance_plot.view, 1, 1, 1, 1)
        self.main_layout.addWidget(self.nucl_content_plot, 1, 2, 1, 1)
        self.main_layout.addWidget(self.img_display, 1, 3, 1, 1)
        self.setLayout(self.main_layout)
    
    def clear(self):
        self.distance_plot.clear()
        self.solutions = []
        img = os.path.join(config.UTILS, 'draw_rna', 'blank.svg')
        self.img_display.load(img)
        self.distance_plot.crosshair.set_pos(0, 0)
        self.gcau()

    def new_solution(self, solution, color_id=0):
        color_id = color_id % 6
        colors = {0:'g', 1:'b', 2:'r', 3:'y', 4:'colors', 5:'k', 6:'m'}
        self.solutions.append(solution)
        color = pg.mkBrush(colors[color_id])
        self.distance_plot.addPoints([solution.hd], [solution.md],brush=color)
        self.highlight(len(self.solutions)-1, cross=False)

    def highlight(self, index=None, cross=True):
        if index is not None and cross:
            self.distance_plot.point_selected(index)
        self.gcau(index)
        self.load_image(index)

    def gcau(self, ind=None):
        if ind is None:
            heights = [0, 0, 0, 0]
        else:
            percentage = self.solutions[ind].gcau_content()
            heights = [percentage['G'], percentage['C'], percentage['A'], percentage['U']]
        self.nucl_content_plot.bar_plot(heights=heights, colors=['r','g','y','b'])

    def load_image(self, index):
        solution = self.solutions[index]
        draw(solution.folded_structure, solution.string, col=None, mismatches=solution.mismatch_indices, name='failed')
        img = os.path.join(config.UTILS, 'draw_rna', 'failed.svg')
        self.img_display.load(img)

class DirectoryComboBox(QtWidgets.QWidget):
    def __init__(self, parent, directory, name, icon, triggers):
        super(DirectoryComboBox, self).__init__(parent=parent)
        self.par = parent
        self.icon = icon
        self.name = name
        self.fsm  = QtWidgets.QFileSystemModel()
        self.selection = QtWidgets.QComboBox()
        self.set_path(directory)

        self.label = QtWidgets.QLabel(name)
        self.label.setFixedWidth(120*config.scaling[0])
        self.label.setPixmap(QtGui.QPixmap(get_icon(self.icon)))
        self.label.setText(name)
        
        ic = QtWidgets.QLabel()
        ic.setPixmap(QtGui.QPixmap(get_icon(self.icon)))
        ic.setFixedWidth(30*config.scaling[0])

        self.main_layout = QtWidgets.QHBoxLayout()
        self.main_layout.addWidget(ic)
        self.main_layout.addWidget(self.label)
        self.main_layout.addWidget(self.selection)
        self.setLayout(self.main_layout)

    def set_path(self, directory):
        index = self.fsm.setRootPath(directory)
        self.selection.setModel(self.fsm)
        self.selection.setRootModelIndex(index)
        self.selection.setCurrentIndex(0)

    def check_existing(self, value):
        return True if self.selection.findData(value) >= 0 else False


class LabeledComboBox(QtWidgets.QWidget):
    def __init__(self, parent, label, items, selection='Default'):
        super(LabeledComboBox, self).__init__(parent=parent)
        self.label = QtWidgets.QLabel(label)

        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(items)
        # icon = QtGui.QIcon(QtGui.QPixmap(get_icon(selection)))
        # self.combo.insertItem(0, icon, label)
        # self.combo.model().item(0).setEnabled(False)
        # self.combo.setCurrentIndex(0)

        self.main_layout = QtWidgets.QHBoxLayout()
        self.main_layout.addWidget(self.label)
        self.main_layout.addWidget(self.combo)
        self.setLayout(self.main_layout)

class LabeledCheckBox(QtWidgets.QWidget):
    def __init__(self, parent, label, items, selection='Default'):
        super(LabeledCheckBox, self).__init__(parent=parent)