from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
import rusher.settings as settings

class EnvDisplay(QtWidgets.QWidget):
    """
    For displaying the images of the environment (specific to binpacking environments as other environments just call a different kinds of displays)
    """
    def __init__(self, parent):
        super(EnvDisplay, self).__init__(parent=parent)
        self.layout = QtWidgets.QGridLayout()

        # Environment image
        self.view = pg.PlotItem(enableMenu=False)
        self.imageView = pg.ImageView(view=self.view)
        self.imageView.getView().invertY(False)                  # Flip Y axis
        self.imageView.getView().showGrid(True, True)            # Enable grid

        # Bin packing-specific buttons
        self.stack_builder_button = QtWidgets.QPushButton()
        self.stack_builder_button.setText('Show in StackBuilder')
        self.stack_builder_button.clicked.connect(self.show_in_stack_builder)
        self.validate_button = QtWidgets.QPushButton()
        self.validate_button.setText('Validate Stack')
        self.validate_button.clicked.connect(self.validate)
        self.write_xml_button = QtWidgets.QPushButton()
        self.write_xml_button.setText('Write XML')
        self.write_xml_button.clicked.connect(self.write_xml)
        
        self.layout.addWidget(self.imageView, 2, 1, 1, 3)
        self.layout.addWidget(self.stack_builder_button, 1, 1, 1, 1)
        self.layout.addWidget(self.validate_button, 1, 2, 1, 1)
        self.layout.addWidget(self.write_xml_button, 1, 3, 1, 1)
        self.setLayout(self.layout)

    def updateImage(self, image=None):
        """
        Update the image plot
        """
        if image is None and self.parameters['Rendering']['draw_images']:
            if self.parameters['Environment']['draw_supports']:
                self.imageView.setImage(self.env.S.image[:,:,:])
            else:
                self.imageView.setImage(self.env.S.image[:,:,0])

        else: 
            self.imageView.setImage(image)

    def show_in_stack_builder(self):
        pass

    def validate(self):
        pass

    def write_xml(self):
        pass
