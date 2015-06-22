from PyQt4.QtCore import *
from PyQt4.QtGui import *
from enums import *

from WindowInventory import WindowInventory

class SubWindow(QFrame):
    def __init__(self, _parent):
        super(SubWindow, self).__init__(_parent)
        self.parent = _parent
        self.main_widget = None
        # self.setWindowFlags(Qt.Window|Qt.CustomizeWindowHint|Qt.WindowMaximizeButtonHint|Qt.WindowMinimizeButtonHint\
        # |Qt.WindowCloseButtonHint|Qt.FramelessWindowHint)

        # note Qt.Drawer looks completely different on OSX than on Windows.
        # QWindow on the other hand on linux displays all windows in dock widget and behaves stranegely
        # thus the settings below
        # are actually the ones that work on all platforms

        self.setWindowFlags(Qt.Dialog|Qt.CustomizeWindowHint|Qt.WindowMaximizeButtonHint|Qt.WindowMinimizeButtonHint\
        |Qt.WindowCloseButtonHint|Qt.FramelessWindowHint)

    def setWidget(self, widget):
        '''
        Places widget  in the frame's layout
        :param widget:widget to be added to Qframe
        :return:None
        '''
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))
        layout = QBoxLayout(QBoxLayout.TopToBottom)
        layout.addWidget(widget)
        layout.setMargin(0)
        self.main_widget = widget
        self.setLayout(layout)

    def sizeHint(self):
        '''
        returns suggested size for qframe
        :return:QSize
        '''
        return QSize(400, 400)

    def widget(self):
        '''
        main widget displayed in Qframe
        :return: main widget displayed in Qframe
        '''
        return self.main_widget

    def mousePressEvent(self, ev):
        '''
        handler for mouse click event - updates self.parent.lastActiveRealWindow member variable
        :param ev: mousePressEvent
        :return:None
        '''
        self.parent.lastActiveRealWindow = self
        super(SubWindow,self).mousePressEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        '''
        handler for mouse double-click event - updates self.parent.lastActiveRealWindow member variable
        :param ev:  mouseDoubleClickEvent
        :return:None
        '''
        self.parent.lastActiveRealWindow = self
        super(SubWindow, self).mouseDoubleClickEvent(ev)

    # def changeEvent(self, ev):
    #     '''
    #     sets MainArea's lastActiveRealWindow - currently inactive
    #     :param ev: QEvent
    #     :return:None
    #     '''
    #
    #     return
        # if ev.type() == QEvent.ActivationChange:
        #     if self.isActiveWindow():
        #         print 'will activate ', self
        #         self.parent.lastActiveRealWindow = self
        #
        # super(DockSubWindow,self).changeEvent(ev)

    def closeEvent(self, ev):
        '''
        handler for close event event - removes sub window from inventory
        :param ev:  closeEvent
        :return:None
        '''
        self.parent.win_inventory.remove_from_inventory(self)



class MainArea(QWidget):
    def __init__(self, stv,  ui ):

        self.MDI_ON = False

        self.stv = stv # SimpleTabView
        self.UI = ui # UserInterface

        self.win_inventory = WindowInventory()

        self.lastActiveRealWindow = None # keeps track of the last active real window
        self.last_suggested_window_position = QPoint(0,0)


    def suggested_window_position(self):
        '''
        returns suggested position of the next window
        :return:QPoint - position of the next window
        '''

        rec = QApplication.desktop().screenGeometry()

        if self.last_suggested_window_position.x() == 0 and self.last_suggested_window_position.y() == 0:

            self.last_suggested_window_position = QPoint(rec.width()/5, rec.height()/5)
            return self.last_suggested_window_position
        else:
            from random import randint
            self.last_suggested_window_position = QPoint(randint(rec.width()/5, rec.width()/2), randint(rec.height()/5, rec.height()/2))
            return self.last_suggested_window_position

    def addSubWindow(self, widget):
        '''
        adds subwindow containing widget to the player
        :param widget: widget to be added to sub windows
        :return:None
        '''

        import Graphics
        print 'INSTANCE OF GraphicsFrameWidget =  ', isinstance(widget, Graphics.GraphicsFrameWidget.GraphicsFrameWidget)
        obj_type = 'other'
        if isinstance(widget, Graphics.GraphicsFrameWidget.GraphicsFrameWidget):
            obj_type = GRAPHICS_WINDOW_LABEL

        elif isinstance(widget, Graphics.PlotFrameWidget.PlotFrameWidget):
            obj_type = PLOT_WINDOW_LABEL

        window_name = obj_type + ' ' + str(self.win_inventory.get_counter())

        subWindow = self.createSubWindow(name=window_name) # sub window
        self.setupSubWindow(subWindow, widget, self.trUtf8(window_name))

        # inserting widget into dictionary
        self.win_inventory.add_to_inventory(obj = subWindow, obj_type=obj_type)

        return subWindow

    def tileSubWindows(self):
        '''
        dummy function to make conform to QMdiArea API
        :return: None
        '''
        pass

    def cascadeSubWindows(self):
        '''
        dummy function to make conform to QMdiArea API
        :return: None
        '''
        pass

    def activeSubWindow(self):
        '''
        returns last active subwindow
        :return: SubWindow object
        '''
        print 'returning lastActiveRealWindow=', self.lastActiveRealWindow
        return self.lastActiveRealWindow

    def setActiveSubWindow(self, win):
        '''
        Activates subwindow win
        :param: win - SubWindow object
        :return: None
        '''
        win.activateWindow()
        self.lastActiveRealWindow = win

    def subWindowList(self):
        '''
        returns list of all open subwindows
        :return: python list of SubWindow objects
        '''
        return self.win_inventory.values()

    def createSubWindow(self, name):
        '''
        Creates SubWindow with title specified using name parameter
        :param: name -  subwindow title
        :return: SubWindow object
        '''

        sub_window = SubWindow(self)
        sub_window .setObjectName(name)
        return sub_window

    def setupSubWindow(self, sub_window, widget, caption):
        '''
        Configures subwindow by placing widget in to qframe layout, setting window title (caption)
        and showing subwindow
        :param: sub_window - SubWindow object
        :param: widget - widget to be placed into sub_window
        :param: caption - subwindow title
        :return: None
        '''

        if caption is None:
            caption = QString()

        sub_window.setWindowTitle(caption)
        sub_window.setWidget(widget)
        sub_window.show()


# class DockSubWindow(QDockWidget):
#     def __init__(self, _parent):
#         super(DockSubWindow, self).__init__(_parent)
#         self.parent = _parent
#         self.setAllowedAreas(Qt.NoDockWidgetArea)
#         # self.toggleFcn = None
#     # def setToggleFcn(self, fcn):self.toggleFcn = fcn
#
#     def mousePressEvent(self, ev):
#         self.parent.lastActiveRealWindow = self
#         # self.parent.lastClickedRealWindow = self
#         super(DockSubWindow,self).mousePressEvent(ev)
#
#     def mouseDoubleClickEvent(self, ev):
#         self.parent.lastActiveRealWindow = self
#         # self.parent.lastClickedRealWindow = self
#         super(DockSubWindow,self).mouseDoubleClickEvent(ev)
#
#     def changeEvent(self, ev):
#         '''
#         sets MainArea's lastActiveRealWindow
#         :param ev: QEvent
#         :return:None
#         '''
#
#         return
#         if ev.type() == QEvent.ActivationChange:
#             if self.isActiveWindow():
#                 print 'will activate ', self
#                 self.parent.lastActiveRealWindow = self
#
#         super(DockSubWindow,self).changeEvent(ev)
#
#
#     def closeEvent(self, ev):
#         # print 'DOCK WIDGET CLOSE EVENT'
#         # # print 'self.toggleFcn=', self.toggleFcn
#         # print 'self = ', self
#         # print 'BEFORE self.parent.win_inventory = ',self.parent.win_inventory
#         self.parent.win_inventory.remove_from_inventory(self)
#
#         # print 'AFTER self.parent.win_inventory = ',self.parent.win_inventory
#
#         # self.windowInventoryDict[self.windowInventoryCounter] = dockWidget
#         # if self.toggleFcn: self.toggleFcn(False)
#         # Configuration.setSetting(str(self.objectName(), False)
#
#
# class MainArea(QWidget):
#     def __init__(self, stv,  ui ):
#
#         self.MDI_ON = False
#
#         self.stv = stv # SimpleTabView
#         self.UI = ui # UserInterface
#
#         self.win_inventory = WindowInventory()
#
#         self.lastActiveRealWindow = None # keeps track of the last active real window
#
#         # self.lastClickedRealWindow = None # keeps track of the last clicked real window
#
#
#     def addSubWindow(self, widget):
#
#
#         import Graphics
#         print 'INSTANCE OF GraphicsFrameWidget =  ', isinstance(widget, Graphics.GraphicsFrameWidget.GraphicsFrameWidget)
#         obj_type = 'other'
#         if isinstance(widget, Graphics.GraphicsFrameWidget.GraphicsFrameWidget):
#             # obj_type = 'graphics'
#             obj_type = GRAPHICS_WINDOW_LABEL
#         elif isinstance(widget, Graphics.PlotFrameWidget.PlotFrameWidget):
#             obj_type = PLOT_WINDOW_LABEL
#             # obj_type = 'plot'
#
#         window_name = obj_type + ' ' + str(self.win_inventory.get_counter())
#         dockWidget = self.createDockWindow(name=window_name) # graphics dock window
#         self.setupDockWindow(dockWidget, Qt.NoDockWidgetArea, widget, self.trUtf8(window_name))
#         # dockWidget = self.createDockWindow(name="Graphincs Window") # graphics dock window
#         # self.setupDockWindow(dockWidget, Qt.NoDockWidgetArea, widget, self.trUtf8("Graphincs Window"))
#
#         # inserting widget into dictionary
#         # self.win_inventory.add_to_inventory( obj = dockWidget)
#         self.win_inventory.add_to_inventory(obj=dockWidget, obj_type=obj_type)
#
#         # self.windowInventoryDict[self.windowInventoryCounter] = dockWidget
#         #
#         # self.windowInventoryCounter += 1
#
#         return dockWidget
#
#     def tileSubWindows(self): pass
#
#     def cascadeSubWindows(self): pass
#
#     # def clickedSubWindow(self):
#     #     return self.lastClickedRealWindow
#
#     def activeSubWindow(self):
#         print 'returning lastActiveRealWindow=', self.lastActiveRealWindow
#         return self.lastActiveRealWindow
#
#     def setActiveSubWindow(self, win):
#         win.activateWindow()
#
#         self.lastActiveRealWindow = win
#
#         pass
#
#     def subWindowList(self):
#         return self.win_inventory.values()
#         # return self.windowInventoryDict.values()
#
#     def createDockWindow(self, name):
#         """
#         Private method to create a dock window with common properties.
#
#         @param name object name of the new dock window (string or QString)
#         @return the generated dock window (QDockWindow)
#         """
#         # dock = QDockWidget(self)
#         # dock = QDockWidget(self)
#         dock = DockSubWindow(self)
#         dock.setObjectName(name)
#         #dock.setFeatures(QDockWidget.DockWidgetFeatures(QDockWidget.AllDockWidgetFeatures))
#         return dock
#
#     def setupDockWindow(self, dock, where, widget, caption):
#         """
#         Private method to configure the dock window created with __createDockWindow().
#
#         @param dock the dock window (QDockWindow)
#         @param where dock area to be docked to (Qt.DockWidgetArea)
#         @param widget widget to be shown in the dock window (QWidget)
#         @param caption caption of the dock window (string or QString)
#         """
#         if caption is None:
#             caption = QString()
#
#         dock.setFloating(True)
#         # print 'self.parent = ', self.parent
#         self.UI.addDockWidget(where, dock)
#         dock.setWidget(widget)
#         dock.setWindowTitle(caption)
#         dock.show()
