from PyQt4.QtCore import *
from PyQt4.QtGui import *

class MainArea(QWidget):
    def __init__(self, stv,  ui ):

        self.MDI_ON = False

        self.stv = stv # SimpleTabView
        self.UI = ui # UserInterface
        self.windowInventoryCounter = 0

        self.windowInventoryDict = {}

    def addSubWindow(self, widget):

        # gfw = GraphicsFrameWidget(parent=None, originatingWidget=self)
        # self.mainGraphicsWindow = gfw

        dockWidget = self.createDockWindow(name="Graphincs Window") # graphics dock window
        self.setupDockWindow(dockWidget, Qt.NoDockWidgetArea, widget, self.trUtf8("Graphincs Window"))

        # inserting widget into dictionary
        self.windowInventoryDict[self.windowInventoryCounter] = dockWidget

        self.windowInventoryCounter += 1

        return dockWidget

    def tileSubWindows(self): pass

    def cascadeSubWindows(self): pass

    def setActiveSubWindow(self, win):
        pass

    def subWindowList(self):
        return self.windowInventoryDict.values()

    def createDockWindow(self, name):
        """
        Private method to create a dock window with common properties.

        @param name object name of the new dock window (string or QString)
        @return the generated dock window (QDockWindow)
        """
        # dock = QDockWidget(self)
        dock = QDockWidget(self)
        dock.setObjectName(name)
        #dock.setFeatures(QDockWidget.DockWidgetFeatures(QDockWidget.AllDockWidgetFeatures))
        return dock

    def setupDockWindow(self, dock, where, widget, caption):
        """
        Private method to configure the dock window created with __createDockWindow().

        @param dock the dock window (QDockWindow)
        @param where dock area to be docked to (Qt.DockWidgetArea)
        @param widget widget to be shown in the dock window (QWidget)
        @param caption caption of the dock window (string or QString)
        """
        if caption is None:
            caption = QString()

        dock.setFloating(True)
        print 'self.parent = ', self.parent
        self.UI.addDockWidget(where, dock)
        dock.setWidget(widget)
        dock.setWindowTitle(caption)
        dock.show()
