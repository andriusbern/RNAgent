from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtSvg import *
import pyqtgraph as pg

import os, time, shutil, sys, re
import traceback, random
import numpy as np

import rlif.environments
from rlif.rna import Dataset, DotBracket, Solution, load_sequence
from rlif.learning import Trainer, get_parameters
from rlif.rna import colorize_nucleotides, highlight_mismatches, colorize_motifs
from rlif.utils import draw
from rlif.learning import RLIF

from rlif.settings import ConfigManager as config

import RNA, forgi
param_files = [
    'rna_turner1999.par',
    'rna_turner2004.par',
    'rna_andronescu2007.par',
    'rna_langdon2018.par']

def set_vienna_param(param, value):
    setattr(RNA.cvar, param, value)

def set_vienna_params(n):

    params = os.path.join(config.MAIN_DIR, 'utils', 'parameters', param_files[n])
    RNA.read_parameter_file(params)

def get_icon(name):
    icon_path = os.path.join(config.ICONS, name+'.svg')
    return icon_path

def clicked(item, points):
    indexes = []
    for p in points:
        p = p.pos()
        x, y = p.x(), p.y()
        lx = np.argwhere(item.data['x'] == x)
        ly = np.argwhere(item.data['y'] == y)
        i = np.intersect1d(lx, ly).tolist()
        indexes += i
    indexes = list(set(indexes))
    return indexes[0]

class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)

class Worker(QThread):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()   
    
    @pyqtSlot()
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

class MainWidget(QWidget):
    def __init__(self, parent):
        super(MainWidget, self).__init__(parent=parent)
        self.par = parent
        self.load = 1
        self.timer = QTimer()
        self.target = ''
        self.solution = None
        self.nucleotides = None
        self.iter = 0
        self.total_iter = 0
        self.start = 0
        self.sequences = []
        self.solutions = []
        self.failed = []
        self.targets = []
        self.dataset = 'eterna'
        self.create_layout()
        self.testing = False
        self.current_target = 0
        self.running = False

        self.threadpool = QThreadPool()
        self.model = RLIF()
        self.fold_thread = Worker(self.model.single_run)
        self.fold_thread.signals.result.connect(self.process_output)
        self.step_thread = Worker(self.model.single_step)
        self.step_thread.signals.result.connect(self.process_step)

        self.vfold_thread = Worker(self.vff)
        self.vfold_thread.signals.result.connect(self.process_output)

    def clear_all(self):
        self.solution_viz.clear()
        self.failed_viz.clear()
        self.results_table.clear()
        self.results_table.clearContents()
        self.failed_table.clearContents()
        for row in reversed(range(self.results_table.rowCount())):
            self.results_table.removeRow(row)
        for row in reversed(range(self.failed_table.rowCount())):
            self.failed_table.removeRow(row)
        self.failed_table.clear()
        for row in reversed(range(self.targets_table.rowCount())):
            self.targets_table.removeRow(row)

        self.total_iter = 0
        self.start = 0
        self.sequences = []
        self.solutions = []
        self.failed = []
        self.targets = []
        self.update_progress()
        self.target = None
        self.nucleotides = None

    def create_layout(self, redo=False):

        self.sequence_input = QLineEdit(self)
        self.sequence_input.setMinimumWidth(1000)
        self.sequence_input.setPlaceholderText('Enter a nucleotide sequence (A/C/G/U) or a secondary RNA structure in dot-bracket format.')
        self.sequence_input.textChanged.connect(self.parse_input)
        self.sequence_input.returnPressed.connect(self.update_statistics)

        # Progress
        self.progress_view = QWidget()
        prlayout = QGridLayout(self.progress_view)
        self.progress_bar = QProgressBar()
        self.progress_display = QTextEdit(self)
        self.progress_display.setText('Progress:')
        self.targets_table = TargetTable('Targets', self, None)
        self.targets_table.setFixedWidth(200)
        # self.pd = QTextEdit(self)
        # self.pd.setText('Idle.')
        self.progress_display.setEnabled(False)
        prlayout.addWidget(self.progress_display, 1, 1, 1, 1)
        prlayout.addWidget(self.progress_bar, 2, 1, 1, 1)
        self.progress_view.setFixedHeight(200)
        
        # Sequence display
        self.img_tabs = QTabWidget()
        self.img_disp = QSvgWidget()
        self.cnt_disp = QSvgWidget()
        self.img_tabs.addTab(self.img_disp, 't1')
        self.img_tabs.setTabText(0, 'MFE')
        self.img_tabs.addTab(self.cnt_disp, 't2')
        self.img_tabs.setTabText(1, 'Centroid')
        self.img_tabs.setFixedSize(400,400)
        self.img_tabs.currentChanged.connect(self.reload)
        self.img_disp.setStatusTip('Minimum Free Energy (MFE) structure.')
        self.cnt_disp.setStatusTip('Centroid of the ensemble.')

        self.target_statistics = QTextEdit(self)
        self.target_statistics.setText('Target Statistics:')
        self.target_statistics.setFixedHeight(150)
        self.sequence_statistics = QTextEdit(self)
        # self.sequence_statistics.setText()
        self.sequence_statistics.setFixedHeight(150)
                
        # Valid solution tab
        self.results_table = SolutionTable('None', self, None)
        self.solution_viz = SolutionWidget('Plots', self, None)
        self.valid_view = QWidget()
        vallay = QVBoxLayout(self.valid_view)
        vallay.addWidget(self.solution_viz)
        vallay.addWidget(self.results_table)
        self.results_table.setMinimumHeight(225)

        # Failed solution tab
        self.failed_viz = FailedWidget('Failed Solutions', self, None)
        self.failed_table = FailedTable('None', self, None)
        self.failed_view = QWidget()
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
        self.fold_button  = ToggleButton(self, ['Run', 'Stop'], 'loop', ['Run RLIF to generate nucleotide sequences for the target structure.', 'Stop.'])
        self.vfold_button = ToggleButton(self, ['RNAinverse', 'Stop'], 'vloop', ['Use RNAinverse.', 'Stop.'])
        self.test_button = ToggleButton(self, ['Test', 'Stop'], 'test', ['Run RLIF one nucleotide at a time.', 'Stop.'])
        self.load_button  = ClickButton(self, 'Load', [self.load_seqs], status='Load log files.')
        self.save_button  = ClickButton(self, 'Save', [self.save_seqs], status='Save solutions.')
        self.clear_button = ClickButton(self, 'Clear', [self.clear_all], status='Clear all solutions.')
        self.random_button = ClickButton(self, 'Random', [self.random], 'Random nucleotide sequence.')
        self.random_button.setFixedWidth(100)
        self.buttons = QGroupBox('Control')
        self.buttons.setFixedSize(400, 230)

        self.dataset_selection = DirectoryComboBox(
            parent=self, 
            directory=config.DATA,
            name='Dataset', 
            icon='Dataset', 
            triggers=[self.change_dataset])
        self.dataset_selection.selection.currentTextChanged.connect(self.change_dataset)

        # self.data_selection = DirectoryComboBox(
        #     parent=self, 
        #     directory=os.path.join(config.DATA, self.dataset), 
        #     name='Sample', 
        #     icon='Data', 
        #     triggers=[self.load_file])
        # self.data_selection.selection.currentIndexChanged.connect(self.load_file)
        self.load_dataset_button = ClickButton(self, 'Load Dataset', [self.load_dataset], 'Load dataset.')
        self.parameters = ParameterContainer('Parameters', self)

        self.control_group = QWidget()
        lays = QVBoxLayout(self.control_group)
        lays.addWidget(self.buttons)
        lays.addWidget(self.parameters)
        self.control_group.setFixedWidth(400)
        
        btnl = QGridLayout(self.buttons)
        btnl.addWidget(self.fold_button, 1, 1, 1, 1)
        btnl.addWidget(self.test_button, 1, 2, 1, 1)
        btnl.addWidget(self.vfold_button, 2, 1, 1, 1)
        btnl.addWidget(self.clear_button, 2, 2, 1, 1)
        btnl.addWidget(self.save_button, 3, 1, 1, 1)
        btnl.addWidget(self.load_button, 3, 2, 1, 1)
        btnl.addWidget(self.dataset_selection, 4, 1, 1, 2)
        btnl.addWidget(self.load_dataset_button, 5, 1, 1, 2)

        self.stats = QWidget()
        stl = QHBoxLayout(self.stats)
        stl.addWidget(self.target_statistics)
        stl.addWidget(self.sequence_statistics)

        self.lay = QGridLayout(self)
        # Layout
        self.lay.addWidget(self.sequence_input, 1, 1, 1, 3)
        self.lay.addWidget(self.random_button, 1, 4, 1, 1)
        self.lay.addWidget(self.stats, 2, 1, 1, 4)
        self.lay.addWidget(self.control_group, 3, 1, 4, 1)
        self.lay.addWidget(self.progress_view, 3, 2, 1, 3)
        self.lay.addWidget(self.img_tabs, 1, 5, 3, 1)
        self.lay.addWidget(self.tabs, 4, 2, 3, 4)
        self.lay.addWidget(self.targets_table, 1, 6, 6, 1)
        
        # self.lay.addWidget(self.parameters, 4, 1, 3, 1)

        self.parameter_changed()
        self.sequence_input.setText(self.target)
        self.parse_input()
    
    def load_dataset(self):
        self.clear_all()
        self.to_load = [1, 100] if self.dataset != 'rfam_modena' else [1, 29]
        self.timer = QTimer()
        self.timer.timeout.connect(self.load_file)
        self.timer.start(100)

    def reload(self, num):
        self.load_image()

    def change_dataset(self, data):
        self.dataset = data
        # self.data_selection.set_path(os.path.join(config.DATA, self.dataset))

    def random(self):
        mapping = {0:'A', 1:'C', 2:'G', 3:'U'}
        length = random.randint(50, 350)
        sequence = ''.join([mapping[random.randint(0,3)] for x in range(length)])
        self.nucleotides = sequence
        self.nucl_fold(True, reset=True)
        # self.sequence_input.setText(sequence)

    def loop(self, status):
        if status:
            self.load += 1
            self.model.prep(self.target)
            if self.testing:
                self.model.envs[0].reset()
                self.progress_bar.setRange(0, len(self.target))
            self.start = time.time()
            self.timer.timeout.connect(self.fold)
            self.timer.start(1)
        else:
            self.reset()

    def test(self, status):
        self.testing = status
        if status:
            self.model.prep(self.target)
            self.model.envs[0].reset()
            self.progress_bar.setRange(0, len(self.target))
            self.start = time.time()
            self.timer.timeout.connect(self.fold)
            self.timer.start(1)
        else:
            self.reset()

    def fold(self):
        if self.testing:
            self.step_thread.start()
        else:
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

    def process_output(self, solution):
        self.running = False
        if type(solution) is not str and type(solution) is not list:
            solution = solutions = [solution]

        if type(solution[0]) is str:
            if len(solution) > 3:
                solution, hd, t0, t = solution
            else:
                solution, t0, t = solution
            native = 0 if t=='l' else 5
            source = 'learna' if t=='l' else 'RNAinverse'
            conf = self.model.envs[0].config
            target = DotBracket(self.target)
            solution = Solution(
                target=target, 
                config=conf, 
                string=solution, 
                time=time.time()-t0, 
                source=source)
            if solution.hd == 0:
                self.new_solution(solution, native=native)
                self.nucleotides = solution.string
                self.load_image()
            else:
                self.new_failed(solution, native=native)
            self.solution = solution
        else:
            for solution in solutions:
                if solution.hd == 0:
                    self.new_solution(solution, native= self.load)
                    self.solution = solution
                    self.nucleotides = solution.string
                    self.load_image()
                else:
                    self.new_failed(solution, native=self.load)
        self.update_progress()
        self.update_statistics(render=False, new_target=False)
        
    def process_step(self, solution):
        # print(solution)
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
        self.load_image()
        if done:
            self.reset()
            self.process_output(solution)

    def new_solution(self, solution, native=0):
        self.sequences.append(solution)
        self.solutions.append(solution)
        self.results_table.new_solution(solution, native=native)
        self.solution_viz.new_solution(solution, native=native)
        self.targets_table.update(self.current_target, solved=True)

    def new_failed(self, solution, native=0):
        self.sequences.append(solution)
        self.failed.append(solution)
        self.failed_table.new_solution(solution, native=native)
        self.failed_viz.new_solution(solution, native=native)
        self.targets_table.update(self.current_target, solved=False)

    def new_target(self, target, img=None):
        self.targets.append(target)
        self.current_target = len(self.targets) - 1
        self.target = target.seq
        self.targets_table.new_target(target)

    def target_selected(self, index):
        self.current_target = index
        target = self.targets[index]
        self.target = target.seq
        self.nucleotides = self.targets[index].nucleotides
        self.update_statistics(render=True, new_target=False)

    def nucl_fold(self, btn=True, reset=False):
        if self.nucleotides != None:
            target, _ = RNA.fold(self.nucleotides)
            if target != self.target:
                conf = self.model.envs[0].config
                t = DotBracket(target)
                t.nucleotides = self.nucleotides
                self.target = target
                self.solution = Solution(target=t, config=conf, string=self.nucleotides)
                if reset:
                    self.nucleotides = None
                self.update_statistics(True, new_target=btn, target=t)
                
    def load_image(self, seq=None):
        if type(seq) is int:
            self.load_image(None)
        if self.img_disp is self.img_tabs.currentWidget():
            sequence = self.target if seq is None else seq
            disp = self.img_disp
        else:
            if self.solution is not None:
                self.solution.compute_statistics()
                sequence = self.solution.centroid_structure
                disp = self.cnt_disp
        draw(sequence, self.nucleotides, col=None)
        img = os.path.join(config.UTILS, 'draw_rna', 'output.svg')
        disp.load(img)
        return img

    def load_file(self):
        ind = self.to_load[0]
        target = load_sequence(num=int(ind), dataset=self.dataset)
        self.target = target.seq
        self.nucleotides = None
        self.to_load[0] += 1
        if ind == self.to_load[1]:
            self.timer.stop()
            self.timer = QTimer()
        self.update_statistics(new_target=True, target=target)

        
    def parse_input(self, str=None):
        dot_bracket_check = re.compile(r'[^.)()]').search # Regex for dotbrackets
        nucl_check = re.compile(r'[^AUGCaugc]').search
        current_text = self.sequence_input.text().strip()
        if not bool(dot_bracket_check(current_text)) and len(current_text) > 0:
            self.nucleotides = None
            self.target = current_text
            self.update_statistics(new_target=False)
            self.fold_button.setEnabled(True)
            self.vfold_button.setEnabled(True)
        elif not bool(nucl_check(current_text)) and len(current_text) > 0:
            self.nucleotides = current_text.upper()
            self.nucl_fold(False)
        else:
            self.target_statistics.setText('Invalid input.')

    def reset(self):
        self.running = False
        self.timer.stop()
        self.timer = QTimer()
        self.iter = 0
        self.vfold_button.stop()
        self.fold_button.stop()
        self.test_button.stop()
        self.testing = False
        
    def update_statistics(self, render=True, new_target=True, target=None, solution=None):
        
        if target is None:
            target = DotBracket(self.target)
            nucleotides = self.nucleotides
        else:
            self.target = target.seq
            nucleotides = target.nucleotides

        if render:
            img = self.load_image()

        if new_target:
            self.new_target(target)
            target = DotBracket(self.target)

        if solution is not None:
            target = solution.target
            self.target = solution.target.seq
            self.nucleotides = nucleotides = solution.string
            
        
        f, = forgi.load_rna(target.seq)
        stems = len([s for s in f.stem_iterator()])
        stats = '{:25}: {:6} | {:20}: {:3}'.format(
            'Target length', target.len, 'Hairpin loops', target.counter['H'])
        stats += '\n{:25}: {:.2f}% | {:20}: {:3}\n\n'.format(
            'Unpaired Nucleotides', target.percent_unpaired*100, 'Interior loops', target.counter['H'])
        
        ruler = ''.join(['  {:3}'.format(i) for i in range(0, 81, 5)[1:]]) + '\n'
        stats = ruler
        nstats = ruler

        for i in range(len(target.seq)//80+1):
            stats += target.seq[i*80:(i+1)*80] + '\n'
            if nucleotides is not None:
                nstats += nucleotides[i*80:(i+1)*80] + '\n'
        nstats += '\n'+ target.name

        self.target_statistics.setText(stats)
        self.sequence_statistics.setText(nstats)
        
        # except:
        #     pass

    def update_progress(self):
        self.progress_bar.setValue(self.iter)
        text = '{:20}: {:4}/{:4}'.format('Attempt', self.iter, config.ATTEMPTS)
        text += '\n{:20}: {:8}'.format('Total attempts', self.total_iter)
        text += '\n{:20}: {:8}'.format('Correct solutions', len(self.solutions))
        text += '\n{:20}: {:8}'.format('Failed solutions', len(self.failed))
        self.progress_display.setText(text)
    
    def load_seqs(self):
        sources1 = {'RNAinverse':'RNAinverse', 'MODENA':'MODENA', 'NUPACK':'NUPACK', 'antaRNA':'antaRNA', 'rnaMCTS':'rnaMCTS', 'LEARNA':'LEARNA', 'rlif':'rlif', 'rlif*':'rlif','rlfold':'rlif', 'brlfold':'rlif', 'mrlfold':'rlif'}
        sources = {'RNAinverse':0, 'MODENA':1, 'NUPACK':2, 'antaRNA':3, 'rnaMCTS':4, 'LEARNA':5, 'rlif':6}
        name = QFileDialog.getOpenFileName()[0]
        conf = self.model.envs[0].config
        ext = name.split('.')[-1]
        if ext in ['fasta', 'fa']:
            fname = name.split(config.delimiter)[-1].split('.')[0]
            with open(name, 'r') as fasta:
                seq = ''
                for line in fasta.readlines():
                    if line[0] == '>':
                        if seq != '':
                            seq = seq.replace('T', 'U')
                            structure, fe = RNA.fold(seq)
                            target = DotBracket(structure)
                            target.name = name
                            target_nucleotides = seq
                            solution = Solution(target=target, config=conf, string=seq, time=0, source=fname)
                            self.nucleotides = solution.string
                            self.target = target.seq
                            self.new_target(target)
                            self.new_solution(solution, native=self.load)
                            self.update_statistics(new_target=False)

                            seq = ''
                        name = line[1:].strip('\n')
                    else:
                        seq += line.strip('\n')
        else:
            with open(name, 'r') as seqfile:
                seqs = seqfile.readlines()[1:]
                target_string = seqs[0].strip().strip('\n')
                target = DotBracket(target_string)
                self.target = target_string
                self.update_statistics(new_target=True)
                self.sequence_input.setText(target_string)
                for seq in seqs[1:]:
                    seq = seq.strip().strip('\n')
                    seq, t, source = seq.split(' ')
                    source = sources1[source]
                    # if seq != '---':
                    solution = Solution(target=target, config=conf, string=seq, time=float(t), source=source)

                    if solution.hd == 0:
                        self.new_solution(solution, native=sources[source])
                        self.nucleotides = solution.string
                        self.solution = solution
                        self.target = target_string
                        self.update_statistics(new_target=True)
                    else:
                        self.new_failed(solution)
        self.load+= 1
            
    def save_seqs(self):
        save_dir = QFileDialog.getSaveFileName()[0]

        with open(save_dir, 'w') as seqfile:
            seqfile.write('{} valid solutions, {} invalid solutions\n'.format(len(self.solutions), len(self.failed)))
            seqfile.write(self.target+'\n')
            for seq in self.sequences:
                seqfile.write(seq.string + ' {:.2f} '.format(seq.time) + seq.source +'\n')

    def solution_selected(self, index, row=True):
        solution = self.solutions[index]
        self.solution = solution
        if row:
            self.results_table.selectRow(index)
        self.nucleotides = solution.string
        # self.target = solution.target.seq
        self.solution_viz.highlight(index)
        self.update_statistics(new_target=False, solution=solution)
        self.load_image(solution.target.seq)

    def failed_selected(self, index, row=True):
        solution = self.failed[index]
        if row:
            self.failed_table.selectRow(index)
        self.failed_viz.highlight(index)
    
    def parameter_changed(self, parameter=None):
        self.progress_bar.setRange(0, config.ATTEMPTS)
        if parameter in ['temperature', 'dangles', 'noGU', 'no_closingGU', 'uniq_ML']:
            set_vienna_param(parameter, config.get(parameter))
            self.nucl_fold(False)
        if parameter in ['permutation_budget', 'permutation_radius', 'permutation_threshold']:
            conf = self.model.model.env.get_attr('config')[0]
            conf[parameter] = config.get(parameter)
            self.model.model.env.set_attr('config', conf)

class ToggleButton(QPushButton):
    def __init__(self, parent, names, trigger, status=None, own_trigger=False):
        super(ToggleButton, self).__init__(parent=parent)
        self.par = parent
        self.setCheckable(True)
        self.names = names
        self.status = status
        self.setText('   '+ self.names[0])
        if own_trigger:
            self.clicked[bool].connect(getattr(self.par, trigger))
        else:
            self.clicked[bool].connect(getattr(self.par, trigger))
        self.clicked[bool].connect(self.status_change)

        modes = [QIcon.Mode.Normal, QIcon.Mode.Normal, QIcon.Mode.Disabled]
        fns = [QIcon.State.Off, QIcon.State.On, QIcon.State.Off]
        icon = QIcon() # parent=self
        for i,name in enumerate(self.names):
            path = os.path.join(config.ICONS, name+'.svg')
            icon.addPixmap(QPixmap(path), modes[i], fns[i]) #, fns[i]
        self.setIcon(icon)

    def status_change(self, toggled):
        tip = self.names[1] if toggled else self.names[0]
        self.setText('   '+tip)
        self.setStatusTip(self.status[toggled])

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
        path = os.path.join(config.ICONS, name+'.svg')
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
        self.spin_box.setFixedWidth(60)
        self.slider.setValue(self.find_nearest(val))
        self.slider.valueChanged[int].connect(self.value_changed)
        self.spin_box.valueChanged[int].connect(self.update_slider)
        
        name = ' ' * (25 - len(self.translated)) + self.translated + ':'
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
        name = ' ' * (25 - len(name)) + name + ':'
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
        name = '{:25}'.format(config.translate(name) + ': ')
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

        self.fold_params = ParameterComboBox('Energy parameters', self, param_files, self.vienna_params)
        self.checks = CheckboxContainer(
            names=['noGU', 'no_closingGU', 'uniq_ML'],
            parent=self.par,
            fn=self.par.parameter_changed, 
            grid=[2, 2])
        extra = [self.fold_params, self.checks]
        vienna = ['temperature', 'dangles']
        self.vienna_parameters = ParameterGroup('Vienna Parameters', self.par, vienna, extra)
        self.vienna_parameters.setFixedHeight(200)

        rlif = ['TIME', 'N_SOLUTIONS', 'ATTEMPTS', 'WORKERS'] + ['permutation_budget', 'permutation_radius', 'permutation_threshold']
        self.rlif_parameters = ParameterGroup('RL Parameters', self.par, rlif)
        self.lay = QVBoxLayout()
        self.lay.addWidget(self.vienna_parameters)
        self.lay.addWidget(self.rlif_parameters)
        self.setLayout(self.lay)

    def vienna_params(self, param):
        set_vienna_params(param)
        self.par.nucl_fold(False)
        self.par.update_statistics(new_target=False)
        
class SolutionTable(QTableWidget):
    def __init__(self, name, parent, config):
        QTableWidget.__init__(self, 0, 6, parent=parent)
        self.par = parent
        self.setHorizontalHeaderLabels(['ID', 'Nucl seq', 'FE', 'Probability', 't', 'source'])
        self.setColumnWidth(1, 500)
        self.setColumnWidth(0, 50)
        self.currentCellChanged.connect(self.row_selected)
    
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

    def row_selected(self, ind):
        # ind = item.row()
        self.par.solution_selected(ind, row=False)

class FailedTable(QTableWidget):
    def __init__(self, name, parent, config):
        QTableWidget.__init__(self, 0, 7, parent=parent)
        self.par = parent
        self.setHorizontalHeaderLabels(['ID', 'Nucl seq', 'FE', 'HD', 'MD', 't', 'source'])
        self.setColumnWidth(1, 500)
        self.setColumnWidth(0, 50)
        self.currentCellChanged.connect(self.row_selected)

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

    def row_selected(self, ind):
        # ind = item.row()
        # print(ind)
        self.par.failed_selected(ind, row=False)


class TargetTable(QTableWidget):
    def __init__(self, name, parent, config):
        QTableWidget.__init__(self, 0, 4, parent=parent)
        self.par = parent
        self.setHorizontalHeaderLabels(['Seq', 'len', '%S', '%F'])
        # self.setColumnWidth(1, 500)
        self.setColumnWidth(1, 30)
        self.setColumnWidth(2, 30)
        self.setColumnWidth(3, 30)

        self.currentCellChanged.connect(self.row_selected)

    def new_target(self, target):
        count = self.rowCount()
        self.insertRow(count)
        s = 70
        self.setColumnWidth(0, s)
        # QTableWidgetItem().set
        thumb = QSvgWidget(self)
        thumb.setFixedSize(s, s)
        img = os.path.join(config.UTILS, 'draw_rna', 'output.svg')
        thumb.load(img)
        
        self.setCellWidget(count, 0, thumb)
        self.setItem(count, 1, QTableWidgetItem('{}'.format(target.len)))
        self.setItem(count, 2, QTableWidgetItem(str(0)))
        self.setItem(count, 3, QTableWidgetItem(str(0)))
        self.setItem(count, 4, QTableWidgetItem(target.name))
        self.setRowHeight(count, s)

    def row_selected(self, item):
        # print(item)
        if type(item) is int:
            # print(item)
            ind = item
            self.par.target_selected(ind)
        else:
            ind = item.row()
            self.par.target_selected(ind)

    def update(self, row, solved):
        index = 2 if solved else 3
        item = self.item(row, index)
        num = item.setText(str(int(item.text()) + 1))



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
        QGroupBox.__init__(self, parent=parent)
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

    def clear(self):
        self.ed_display.clear()
        self.cnt_display.clear()
        self.prob_display.clear()
        self.gcau(0, vals=[0,0,0,0])
        self.solutions = []
        for cr in self.crs:
            cr.setPos(0, 0)

    def new_solution(self, solution, native=0):
        native = native % 6
        c = {0:'g', 1:'b', 2:'r', 3:'y', 4:'c', 5:'m', 6:'k'}
        self.solutions.append(solution)
        color = pg.mkBrush(c[native])
        self.prob_display.addPoints([solution.fe], [solution.probability],brush=color)
        self.cnt_display.addPoints([solution.fe], [solution.centroid_dist], brush=color)
        self.ed_display.addPoints([solution.fe], [solution.ensemble_defect], brush=color)
        self.highlight(len(self.solutions)-1, cross=False)

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
        
        self.g = pg.BarGraphItem(x=[1], height=[vals[0]], width=.8, brush='r')
        self.c = pg.BarGraphItem(x=[2], height=[vals[1]], width=.8, brush='g')
        self.a = pg.BarGraphItem(x=[3], height=[vals[2]], width=.8, brush='y')
        self.u = pg.BarGraphItem(x=[4], height=[vals[3]], width=.8, brush='b')
        self.gc_view.addItem(self.g)
        self.gc_view.addItem(self.c)
        self.gc_view.addItem(self.a)
        self.gc_view.addItem(self.u)

    def update_image(self):
        pass

    def update_graphs(self):
        pass

    def highlight(self, index, cross=True):
        if cross:
            for cr in self.crs:
                cr.setPos(cr.graph.data[index][0], cr.graph.data[index][1])
        self.gcau(index)

    def solution_selected(self, solution, points):
        ind = clicked(solution, points)
        self.par.solution_selected(ind)

class FailedWidget(QGroupBox):
    def __init__(self, name, parent, config):
        QGroupBox.__init__(self, parent=parent)
        self.par = parent
        self.image_view = pg.ImageView()
        self.image_view.ui.menuBtn.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.getView().getViewBox().setBackground(None)
        self.image_view.getHistogramWidget().setBackground(None)
        self.solutions = []
        self.mismatches = None

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

        self.img_disp = QSvgWidget()
        self.img_disp.setFixedSize(300, 300)
        
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
        self.lay.setColumnMinimumWidth(3, 380)
        self.setLayout(self.lay)
    
    def clear(self):
        self.dist_display.clear()
        img = os.path.join(config.UTILS, 'draw_rna', 'blank.svg')
        self.img_disp.load(img)
        self.gcau(0, vals=[0,0,0,0])
        self.solutions = []
        for cr in self.crs:
            cr.setPos(0, 0)

    def new_solution(self, solution, native=0):
        native = native % 6
        c = {0:'g', 1:'b', 2:'r', 3:'y', 4:'c', 5:'k', 6:'m'}
        self.solutions.append(solution)
        color = pg.mkBrush(c[native])
        self.dist_display.addPoints([solution.hd], [solution.md],brush=color)
        self.highlight(len(self.solutions)-1, cross=False)

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

    def highlight(self, index, cross=True):
        if cross:
            for cr in self.crs:
                cr.setPos(cr.graph.data[index][0], cr.graph.data[index][1])
        self.load_image(index)
        self.gcau(index)

    def solution_selected(self, solution, points):
        ind = clicked(solution, points)
        self.par.failed_selected(ind)

    def load_image(self, index):
        solution = self.solutions[index]
        draw(solution.folded_structure, solution.string, col=None, mismatches=solution.mismatch_indices, name='failed')
        img = os.path.join(config.UTILS, 'draw_rna', 'failed.svg')
        self.img_disp.load(img)


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
        self.label.setPixmap(QPixmap(get_icon(self.icon)))
        self.label.setText(name)
        
        ic = QLabel()
        ic.setPixmap(QPixmap(get_icon(self.icon)))
        ic.setFixedWidth(30)

        self.lay = QHBoxLayout()
        self.lay.addWidget(ic)
        self.lay.addWidget(self.label)
        self.lay.addWidget(self.selection)
        self.setLayout(self.lay)

    def set_path(self, directory):
        index = self.fsm.setRootPath(directory)
        self.selection.setModel(self.fsm)
        self.selection.setRootModelIndex(index)
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