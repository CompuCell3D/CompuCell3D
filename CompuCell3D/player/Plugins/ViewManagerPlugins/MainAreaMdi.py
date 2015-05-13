from PyQt4.QtCore import *
from PyQt4.QtGui import *

from WindowInventory import WindowInventory
from enums import *


class MainArea(QMdiArea):
    def __init__(self, stv,  ui ):
        self.MDI_ON = True

        self.stv = stv # SimpleTabView
        self.UI = ui # UserInterface

        QMdiArea.__init__(self, ui)
        # QMdiArea.__init__(self, parent)

        self.scrollView = QScrollArea(self)
        self.scrollView.setBackgroundRole(QPalette.Dark)
        self.scrollView.setVisible(False)

        #had to introduce separate scrollArea for 2D and 3D widgets. for some reason switching graphics widgets in Scroll area  did not work correctly.
        self.scrollView3D = QScrollArea(self)
        self.scrollView3D.setBackgroundRole(QPalette.Dark)
        self.scrollView3D.setVisible(False)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        pass
        # self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.win_inventory = WindowInventory()
        self.lastActiveRealWindow = None # keeps track of the last active real window

    def suggested_window_position(self):
        '''
        returns suggested position of the next window. For MainAreaMdi it returns QPoint(-1,-1)
        indicating that client code shuld use QMdiArea functionality to place windows
        :return:QPoint - position of the next window
        '''

        return QPoint(-1, -1)

    def addSubWindow(self, widget):

        import Graphics
        obj_type = 'other'
        if isinstance(widget, Graphics.GraphicsFrameWidget.GraphicsFrameWidget):
            # obj_type = 'graphics'
            obj_type = GRAPHICS_WINDOW_LABEL
        elif isinstance(widget, Graphics.PlotFrameWidget.PlotFrameWidget):
            obj_type = PLOT_WINDOW_LABEL
            # obj_type = 'plot'

        window_name = obj_type + ' ' + str(self.win_inventory.get_counter())

        mdi_sub_window = QMdiArea.addSubWindow(self, widget)

        mdi_sub_window.setWindowTitle(window_name)

        self.win_inventory.add_to_inventory(obj=mdi_sub_window, obj_type=obj_type)

        return mdi_sub_window
