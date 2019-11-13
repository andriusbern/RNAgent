from PyQt5 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

from rlif.interface import MainWidget
import sys, os


class Window(QtWidgets.QMainWindow):
    def __init__(self, parent):
        super(Window, self).__init__()
        self.setWindowTitle('RLIF: RNA Inverse folding')
        self.dimensions = [50, 400, 1600, 850]
        self.setGeometry(*self.dimensions)
        self.par = parent
        self.settings = None
        
        exit_action = QtWidgets.QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.close)

        help_action = QtWidgets.QAction('Haelp', self)
        help_action.setShortcut('F1')
        help_action.setStatusTip('Help')
        help_action.triggered.connect(self.showdialog)

        style_action = QtWidgets.QAction('Dark UI', self)
        style_action.setShortcut('F2')
        style_action.setStatusTip('Set dark UI')
        style_action.triggered.connect(self.mode)

        self.status_bar = self.statusBar()
        
        menu = self.menuBar()
        file = menu.addMenu('&File')
        actions = [exit_action, style_action]
        file.addActions(actions)
        
        help_menu = menu.addMenu('&Help')
        help_menu.addAction(help_action)

        # toolbar = self.addToolBar('Exit')
        # toolbar.addActions(actions)

        self.main_widget = MainWidget(self)
        self.setCentralWidget(self.main_widget)
        self.show()

    def display_help(self):
        msg = QtWidgets.QMessageBox()
        msg.setText('Nice.')

    def showdialog(self):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)

        msg.setText("RLIF: RNA design tool")
        msg.setWindowTitle("RLIF")
        msg.setDetailedText("Andrius Bernatavicius, 2019")
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        retval = msg.exec_()

    def settings_menu(self):
        self.settings.setGeometry(QtCore.QRect(1500, 600, 600, 400))
        self.settings.show()
        self.settings.raise_()
    
    def mode(self):
        self.par.set_style()


class RLIFGUI(QtWidgets.QApplication):
    def __init__(self, *args):
        self.setFont(QtGui.QFont('Monospace', 8))
        QtWidgets.QApplication.__init__(self, *args)
        self.setStyle('QtCurve')
        self.window = Window(self)
        self.window.show()

    def closeEvent( self ):
        self.exit(0)
    
    def set_style(self):
        self.set_style('Breeze')
        # pg.setConfigOption('background', 'w')
    
    def main(self, args):
        self.window.show()
        self.window.raise_()
        sys.exit(self.exec_())
    
if __name__ == "__main__":
    interface = App([])
    interface.main(sys.argv)
    