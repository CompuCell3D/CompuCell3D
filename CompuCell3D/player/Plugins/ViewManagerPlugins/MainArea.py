from PyQt4.QtCore import *
from PyQt4.QtGui import *
from enums import *

from WindowInventory import WindowInventory


class DockSubWindow(QDockWidget):
    def __init__(self, _parent):
        super(DockSubWindow, self).__init__(_parent)
        self.parent = _parent
        # self.toggleFcn = None
    # def setToggleFcn(self, fcn):self.toggleFcn = fcn

    def changeEvent(self, ev):
        '''
        sets MainArea's lastActiveRealWindow
        :param ev: QEvent
        :return:None
        '''
        if ev.type() == QEvent.ActivationChange:
            if self.isActiveWindow():
                self.parent.lastActiveRealWindow = self

        super(DockSubWindow,self).changeEvent(ev)


    def closeEvent(self, ev):
        print 'DOCK WIDGET CLOSE EVENT'
        # print 'self.toggleFcn=', self.toggleFcn
        print 'self = ', self
        print 'BEFORE self.parent.win_inventory = ',self.parent.win_inventory
        self.parent.win_inventory.remove_from_inventory(self)

        print 'AFTER self.parent.win_inventory = ',self.parent.win_inventory

        # self.windowInventoryDict[self.windowInventoryCounter] = dockWidget
        # if self.toggleFcn: self.toggleFcn(False)
        # Configuration.setSetting(str(self.objectName(), False)


class MainArea(QWidget):
    def __init__(self, stv,  ui ):

        self.MDI_ON = False

        self.stv = stv # SimpleTabView
        self.UI = ui # UserInterface

        self.win_inventory = WindowInventory()

        self.lastActiveRealWindow = None # keeps track of the last active real window

        # self.windowInventoryCounter = 0
        #
        # self.windowInventoryDict = {}

    def addSubWindow(self, widget):

        # gfw = GraphicsFrameWidget(parent=None, originatingWidget=self)
        # self.mainGraphicsWindow = gfw
        import Graphics
        print 'INSTANCE OF GraphicsFrameWidget =  ', isinstance(widget, Graphics.GraphicsFrameWidget.GraphicsFrameWidget)
        obj_type = 'other'
        if isinstance(widget, Graphics.GraphicsFrameWidget.GraphicsFrameWidget):
            # obj_type = 'graphics'
            obj_type = GRAPHICS_WINDOW_LABEL
        elif isinstance(widget, Graphics.PlotFrameWidget.PlotFrameWidget):
            obj_type = PLOT_WINDOW_LABEL
            # obj_type = 'plot'

        window_name = obj_type + ' ' + str(self.win_inventory.get_counter())
        dockWidget = self.createDockWindow(name=window_name) # graphics dock window
        self.setupDockWindow(dockWidget, Qt.NoDockWidgetArea, widget, self.trUtf8(window_name))
        # dockWidget = self.createDockWindow(name="Graphincs Window") # graphics dock window
        # self.setupDockWindow(dockWidget, Qt.NoDockWidgetArea, widget, self.trUtf8("Graphincs Window"))

        # inserting widget into dictionary
        # self.win_inventory.add_to_inventory( obj = dockWidget)
        self.win_inventory.add_to_inventory(obj=dockWidget, obj_type=obj_type)

        # self.windowInventoryDict[self.windowInventoryCounter] = dockWidget
        #
        # self.windowInventoryCounter += 1

        return dockWidget

    def tileSubWindows(self): pass

    def cascadeSubWindows(self): pass

    def activeSubWindow(self):
        return self.lastActiveRealWindow

    def setActiveSubWindow(self, win):
        win.activateWindow()
        pass

    def subWindowList(self):
        return self.win_inventory.values()
        # return self.windowInventoryDict.values()

    def createDockWindow(self, name):
        """
        Private method to create a dock window with common properties.

        @param name object name of the new dock window (string or QString)
        @return the generated dock window (QDockWindow)
        """
        # dock = QDockWidget(self)
        # dock = QDockWidget(self)
        dock = DockSubWindow(self)
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
