from PyQt4.QtCore import *
from PyQt4.QtGui import *

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

        self.windowInventoryCounter = 0

        self.windowInventoryDict = {}

    def addSubWindow(self, widget):

        mdiSubwindow = QMdiArea.addSubWindow(self, widget)

        # inserting widget into dictionary
        self.windowInventoryDict[self.windowInventoryCounter] = widget

        self.windowInventoryCounter += 1

        return mdiSubwindow
