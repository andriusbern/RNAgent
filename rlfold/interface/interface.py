from rlfold.baselines import get_parameters, SBWrapper
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWebKitWidgets import QWebView, QUrl
# QWebView.QUrl
class RNAInterface(QtWidgets.QWidget):
    """

    """
    def __init__(self):
        super(RNAInterface, self).__init__()

        self.model = SBWrapper('RnaDesign', 'overnight-69').load_model(2)

    def setup_elements(self):
        self.target_input = QtWidgets.QTextEdit()
        self.target_input.setText('Target DBR sequence')

        self.nucl_input = QtWidgets.QTextEdit()
        self.nucl_input.setText('Nucl sequence')

        self.fold_button = self._create_button(self.fold, 'Fold')
        self.show_button = self._create_button(self.update_html, 'Show')


        self.display = QtGui.
        self.target_panel = self.setup_target()

        self.layout = QtWidgets.QGridLayout()
        self.layout.addWidget(self.target_panel, 1, 1, 1, 1)
        self.layout.addWidget(self.display, 1, 2, 1, 1)


        self.setLayout(self.layout)

    @staticmethod
    def _create_button(function_to_connect, name):
        """
        Initialize a button with text and connect a member function to it
        """
        button = QtWidgets.QPushButton(name)
        button.setText(name)
        button.clicked.connect(function_to_connect)
        return button

    def setup_target(self):
        target_widget = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.target_input, 1, 1, 1, 1)
        layout.addWidget(self.nucl_input, 2, 1, 1, 1)
        layout.addWidget(self.fold_button, 3, 1, 1, 1)
        target_widget.setLayout(layout)

        return target_widget

    def update_html(self):
        pass

