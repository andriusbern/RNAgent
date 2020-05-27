from PySide2 import QtCore, QtWidgets, QtGui
from rlif.interface import MainWidget
import pyqtgraph as pg
import sys, os

class Window(QtWidgets.QMainWindow):
    def __init__(self, parent):
        super(Window, self).__init__()
        self.setWindowTitle('RLIF: RNA Inverse folding')
        self.dimensions = [50, 400, 1600, 850]
        self.setGeometry(*self.dimensions)
        self.par = parent
        self.settings = None

        style_action = QtWidgets.QAction('Dark UI', self)
        style_action.setShortcut('F2')
        style_action.setStatusTip('Set dark UI')
        style_action.triggered.connect(self.mode)

        self.status_bar = self.statusBar()
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
        msg.exec_()

    def settings_menu(self):
        self.settings.setGeometry(QtCore.QRect(1280, 800, 0, 0))
        self.settings.show()
        self.settings.raise_()
    
    def mode(self):
        self.par.set_style()


class RLIFGUI(QtWidgets.QApplication):
    def __init__(self, *args):
        self.setFont(QtGui.QFont('Monospace', 9))
        QtWidgets.QApplication.__init__(self, *args)
        self.setStyle('QtCurve')
        self.window = Window(self)
        self.window.show()

    def closeEvent( self ):
        self.exit(0)
    
    def main(self, args):
        self.window.show()
        self.window.raise_()
        sys.exit(self.exec_())
    
    