from PyQt5 import QtGui, QtCore, QtWidgets


class Leaderboard(QtWidgets.QWidget):
    """
    Contains the elements needed to display the leaderboard of models showing the most successful ones in terms of various metrics
    """
    def __init__(self):
        super(Leaderboard, self).__init__()
        self.layout = QtWidgets.QGridLayout()

        self.setLayout(self.layout)


    def load_stats(self, dictionary):
        """
        
        """

    