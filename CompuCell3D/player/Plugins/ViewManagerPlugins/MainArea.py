from PyQt4.QtCore import *
from PyQt4.QtGui import *
from collections import defaultdict
from enums import *

class Inventory:
    def __init__(self):

        self.inventory_dict = defaultdict(lambda: defaultdict(int))

        # self.inventory_dict = {}
        self.inventory_counter = 0

    def get_counter(self):

        return self.inventory_counter

    def add_to_inventory(self,  obj, obj_type):

        self.inventory_dict[obj_type][self.inventory_counter] = obj
        self.inventory_counter += 1
        print 'self.inventory_dict=',self.inventory_dict

        # self.inventory_dict[self.inventory_counter] = obj
        # self.inventory_counter += 1

    # def remove_from_inventory_by_name(self, obj_name):
    #     try:
    #         del self.inventory_dict[obj_name]
    #         self.inventory_counter -= 1
    #     except KeyError:
    #         pass
    def getWindowsItems(self, win_type):

        for winId, win in self.inventory_dict[win_type].iteritems():
            yield winId, win

    def remove_from_inventory(self, obj):



        winTypeToRemove = None
        winIdToRemove = None

        for winType, winDict in self.inventory_dict.iteritems():
            for winId, win in winDict.iteritems():

                if win == obj:
                    winIdToRemove = winId
                    winTypeToRemove = winType

                    break

        if winIdToRemove is not None and winTypeToRemove is not None:

            try:
                del self.inventory_dict[winTypeToRemove][winIdToRemove]
            except KeyError:
                pass

        print 'AFTER REMOVING WINDOW self.inventory_dict = ', self.inventory_dict
        # import sys
        # sys.exit()
        # obj_name_to_remove = None
        # for key, val in self.inventory_dict.iteritems():
        #
        #     if val == obj:
        #
        #         obj_name_to_remove = key
        #         break
        #
        # if obj_name_to_remove is not None:
        #
        #     try:
        #         del self.inventory_dict[obj_name_to_remove]
        #     except KeyError:
        #         pass



    # def values(self):return self.inventory_dict.values()
    def values(self):
        return [val for dict_val in self.inventory_dict.values() for val in dict_val.values()]

    def __str__(self):
        return self.inventory_dict.__str__()

class DockSubWindow(QDockWidget):
    def __init__(self,_parent):
        super(DockSubWindow, self).__init__(_parent)
        self.parent = _parent
        # self.toggleFcn = None
    # def setToggleFcn(self, fcn):self.toggleFcn = fcn

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

        self.win_inventory = Inventory()

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

    def activeSubWindow1(self): pass

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
