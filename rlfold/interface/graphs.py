from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg


class Graphs(QtWidgets.QWidget):
    """
    Contains all the elements needed for plotting
    """
    def __init__(self, parent):
        super(Graphs, self).__init__(parent=parent)
        self.graphs = []
        self.layout = QtWidgets.QGridLayout()
        
        for i, graph in enumerate(self.graphs):
            self.layout.addWidget(graph, i, 1, 1, 1)

        self.setLayout(self.layout)

    def _create_graph(self, graph_name, title):
        """
        Creates a new graph with a given title
        """
        setattr(self, graph_name, pg.PlotWidget().getPlotItem().setTitle(title))

    def update_graph(self, graph, data):
        graph = getattr(graph, self)
        graph.getPlotItem().plot(clear=True).setData(data)

    def updateReward(self):
        """
        Override in the RL trainer subclass
        Update the reward plot
        """
        self.rewardWidget.getPlotItem().plot(clear=True).setData(self.mean_reward)

    def updateLoss(self, loss):
        """
        Updates the loss of the policy network
        """
        self.lossWidget.getPlotItem().plot(clear=True).setData(loss)

    def updateActions(self, action):
        """
        Updates the mean action value plot
        """
        # self.actionMeans.append(np.true_divide(self.mean_actions.sum(1),(self.mean_actions!=0).sum(1)))

        # for i in range(3):
        #     self.actionWidget.getPlotItem().plot().setData(means[:])
        action[2] = round(action[2])
        x = list(range(len(action)))
        y = action

        self.actionScatter.setData(x, y)