from PyQt5 import QtGui, QtCore, QtWidgets
import rusher.settings as settings
import ast

class Tables(QtWidgets.QWidget):
    """
    Parameter tables
    """
    def __init__(self, parent):
        super(Tables, self).__init__(parent=parent)
        self.layout = QtWidgets.QGridLayout()

        self.parameters = self.parent.parameters
        # Parameter tables
        self.global_parameter_table = QtWidgets.QTableWidget(0, 1)
        self.global_parameter_table.setWindowTitle('Global')
        self.model_parameter_table = QtWidgets.QTableWidget(0, 1)
        self.policy_parameter_table = QtWidgets.QTableWidget(0,1)
        self.environment_parameter_table = QtWidgets.QTableWidget(0,1)
        self.tables = [
            self.model_parameter_table,
            self.global_parameter_table,
            self.policy_parameter_table,
            self.environment_parameter_table
        ]
        self.reset_tables()
        # Tooltips
        self.tooltip_display = QtWidgets.QTextEdit()
        self.tooltip_display.setText('Parameter tooltips')

        self.layout.addWidget(self.policy_parameter_table, 1, 1, 1, 1)
        self.layout.addWidget(self.model_parameter_table, 2, 1, 1, 1)
        self.layout.addWidget(self.global_parameter_table, 3, 1, 1, 1)
        self.layout.addWidget(self.environment_parameter_table, 4, 1, 1, 1)
        self.layout.addWidget(self.tooltip_display, 5, 1, 1, 1)
        
        self.setLayout(self.layout)


    def table_status(self, mode):
        """
        Sets the status of the buttons
        """
        table_status = {'uninitialized': [True, True, True],
                        'initialized':   [True, True, False],
                        'running'    :   [True, True, False],
                        'paused'     :   [True, True, False]
        }
        for table, state in zip(self.tables, table_status[mode]):
            table.setEnabled(state)
        
    def env_selection_changed(self):
        envName = self.envSelectionDropdown.currentText()
        try:
            self.parameters = settings.parameters[envName]
        except:
            self.parameters = settings.parameters['Default']
        self.reset_tables()

    def reset_tables(self):
        """
        Resets the parameter tables based on the environment currently selected
        """
        try:
            for table in self.tables:
                table.itemChanged.disconnect(self.table_changed)
        except:
            pass
        self.set_data(self.model_parameter_table, 'Model')
        self.set_data(self.global_parameter_table, 'Rendering')
        self.set_data(self.policy_parameter_table, 'Policy')
        self.set_data(self.environment_parameter_table, 'Environment', tooltip=False)
        for table in self.tables:
            table.itemChanged.connect(self.table_changed)

    def set_data(self, table, parameters_dict, tooltip=True):
        """
        Sets the table to a dictionary of values
        """
        print(parameters_dict)
        table.setFont(QtGui.QFont("Monospace", 6))
        parameters = self.parameters[parameters_dict]
        headers = []
        for n, key in enumerate(parameters.keys()):
            headers.append(key)
            val = parameters[key]
            val = str(val)
            item = QtWidgets.QTableWidgetItem(val)
            table.insertRow(table.rowCount())
            table.setRowHeight(n, 7)
            table.setItem(n, 0, item)

        if tooltip:
            table.itemPressed.connect(self.show_parameter_tooltip)
        table.setHorizontalHeaderLabels([parameters_dict])
        table.verticalHeader()
        table.horizontalHeader().setResizeMode(0, QtGui.QHeaderView.Stretch)
        table.setColumnWidth(0, 300)
        table.setVerticalHeaderLabels(headers)

    def table_changed(self, item):
        """
        Change global parameters whenever the table is modified
        """
        row = item.row()
        parent = item.tableWidget()
        parameter = parent.verticalHeader().model().headerData(row, QtCore.Qt.Vertical)
        header = parent.horizontalHeader().model().headerData(0, QtCore.Qt.Horizontal)
        data_type = type(self.parameters[header][parameter])
        if data_type is list:
            # ast enables the conversion of list string '[1,2,3]' string to an actual list
            self.parameters[header][parameter] = ast.literal_eval(item.text())
        else:
            self.parameters[header][parameter] = data_type(item.text())

    def show_parameter_tooltip(self, item):
        """
        Upon selecting an item in the table, show a corresponding tooltip for the parameter description
        """
        row = item.row()
        parent = item.tableWidget()
        parameter = parent.verticalHeader().model().headerData(row, QtCore.Qt.Vertical)
        try:
            text = settings.tooltips[parameter]
        except KeyError:
            text = 'No parameter tooltip found.'
            
        self.tooltip_display.setText(str(text))