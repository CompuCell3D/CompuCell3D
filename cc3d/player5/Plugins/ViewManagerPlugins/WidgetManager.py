
from .PlotWindowInterface import PlotWindowInterface
from PyQt5 import QtCore, QtGui, QtWidgets
# # from PyQt5.QtCore import *

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


from . import PlotManagerSetup
import os, Configuration
from cc3d.core.enums import *



class WidgetManager(QtCore.QObject):
    """
    This class will manage creation and destruction of widgets that the user will request from Python Steppables
    """

    newWidgetSignal = QtCore.pyqtSignal(QtCore.QMutex, object,object)

    def __init__(self, _viewManager=None, _plotSupportFlag=False):
        QtCore.QObject.__init__(self, None)
        self.vm = _viewManager
        self.windowMutex = QtCore.QMutex()
        self.signalsInitialized = False
        self.windowList = []


    def initSignalAndSlots(self):
        # since initSignalAndSlots can be called in SimTabView multiple times (after each simulation restart) we have to ensure that signals are connected only once
        # otherwise there will be an avalanche of signals - each signal for each additional simulation run this will cause lots of extra windows to pop up

        if not self.signalsInitialized:
            self.newWidgetSignal.connect(self.processRequestForNewWidget)
            self.signalsInitialized = True
            # self.connect(self,SIGNAL("newPlotWindow(QtCore.QMutex)"),self.processRequestForNewPlotWindow)


    # def getNewPlotWindow(self, obj=None):
    #
    #
    #     if obj is None:
    #         message = "You are most likely using old syntax for scientific plots. When adding new plot window please use " \
    #                   "the following updated syntax:" \
    #                   "self.pW = self.addNewPlotWindow" \
    #                   "(_title='Average Volume And Surface',_xAxisTitle='MonteCarlo Step (MCS)'," \
    #                   "_yAxisTitle='Variables', _xScaleType='linear',_yScaleType='linear')"
    #
    #         raise RuntimeError(message)
    #
    #     self.plotWindowMutex.lock()
    #
    #     self.newPlotWindowSignal.emit(self.plotWindowMutex, obj)
    #     # processRequestForNewPlotWindow will be called and it will unlock drawMutex but before it will finish runnning (i.e. before the new window is actually added)we must make sure that getNewPlotwindow does not return
    #     self.plotWindowMutex.lock()
    #     self.plotWindowMutex.unlock()
    #     return self.plotWindowList[-1]  # returning recently added window
    #
    # def restoreSingleWindow(self, plotWindowInterface):
    #     '''
    #     Restores size and position of a single, just-added plot window
    #     :param plotWindowInterface: an insance of PlotWindowInterface - can be fetchet from PlotFrameWidget using PlotFrameWidgetInstance.plotInterface
    #     :return: None
    #     '''
    #
    #     windows_layout_dict = Configuration.getSetting('WindowsLayout')
    #     # print 'windowsLayoutDict=', windowsLayoutDict
    #
    #     if not windows_layout_dict: return
    #
    #     if str(plotWindowInterface.title) in windows_layout_dict.keys():
    #         window_data_dict = windows_layout_dict[str(plotWindowInterface.title)]
    #
    #         from Graphics.GraphicsWindowData import GraphicsWindowData
    #
    #         gwd = GraphicsWindowData()
    #         gwd.fromDict(window_data_dict)
    #
    #         if gwd.winType != 'plot':
    #             return
    #
    #         plot_window = self.vm.lastActiveRealWindow
    #         plot_window.resize(gwd.winSize)
    #         plot_window.move(gwd.winPosition)
    #         plot_window.setWindowTitle(plotWindowInterface.title)
    #
    # def getPlotWindowsLayoutDict(self):
    #     windowsLayout = {}
    #     from Graphics.GraphicsWindowData import GraphicsWindowData
    #
    #     for winId, win in self.vm.win_inventory.getWindowsItems(PLOT_WINDOW_LABEL):
    #         plotFrameWidget = win.widget()
    #         plotInterface = plotFrameWidget.plotInterface()  # getting weakref
    #         if not plotInterface:
    #             continue
    #
    #         gwd = GraphicsWindowData()
    #         gwd.sceneName = plotInterface.title
    #         gwd.winType = 'plot'
    #         plotWindow = plotInterface.plotWindow
    #         mdiPlotWindow = win
    #         # mdiPlotWindow = self.vm.findMDISubWindowForWidget(plotWindow)
    #         print 'plotWindow=', plotWindow
    #         print 'mdiPlotWindow=', mdiPlotWindow
    #         gwd.winSize = mdiPlotWindow.size()
    #         gwd.winPosition = mdiPlotWindow.pos()
    #
    #         windowsLayout[gwd.sceneName] = gwd.toDict()
    #
    #     return windowsLayout


    def getNewWidget(self, obj_name, obj_data=None):



        self.windowMutex.lock()

        self.newWidgetSignal.emit(self.windowMutex, obj_name, obj_data)
        # processRequestForNewPlotWindow will be called and it will unlock drawMutex but before it will finish runnning (i.e. before the new window is actually added)we must make sure that getNewPlotwindow does not return
        self.windowMutex.lock()
        self.windowMutex.unlock()
        return self.windowList[-1]  # returning recently added window
        # return None


    def addPythonSteeringPanel(self, panel_data=None):
        '''
        callback method to create Steering Panel window with sliders
        :return: {None or mdiWindow}
        '''
        if not self.vm.simulationIsRunning:
            return

        # self.item_data = panel_data
        item_data = panel_data



        print ('THIS IS ADD STEERING PANEL')
        # from steering.SteeringParam import SteeringParam
        from steering.SteeringPanelView import SteeringPanelView
        from steering.SteeringPanelModel import SteeringPanelModel
        from steering.SteeringEditorDelegate import SteeringEditorDelegate

        # self.item_data = []
        # self.item_data.append(SteeringParam(name='vol', val=25, min_val=0, max_val=100, widget_name='slider'))
        # self.item_data.append(
        #     SteeringParam(name='lam_vol', val=2.0, min_val=0, max_val=10.0, decimal_precision=2, widget_name='slider'))
        #
        # self.item_data.append(
        #     SteeringParam(name='lam_vol_enum', val=2.0, min_val=0, max_val=10.0, decimal_precision=2,
        #                   widget_name='slider'))

        self.steering_window = QWidget()
        layout = QHBoxLayout()

        # model = QStandardItemModel(4, 2)

        # cdf = get_data_frame()
        self.steering_model = SteeringPanelModel()
        self.steering_model.update(item_data)
        # model.update_type_conv_fcn(get_types())

        self.steering_table_view = SteeringPanelView()
        self.steering_table_view.setModel(self.steering_model)

        delegate = SteeringEditorDelegate()
        self.steering_table_view.setItemDelegate(delegate)

        layout.addWidget(self.steering_table_view)
        self.steering_window.setLayout(layout)
        self.steering_table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        mdiWindow = self.vm.addSteeringSubWindow(self.steering_window)

        # IMPORTANT show() method needs to be called AFTER creating MDI subwindow
        self.steering_window.show()

        return mdiWindow


    def processRequestForNewWidget(self, _mutex, obj_name, obj_data=None):
        print('obj_name=', obj_name)


        if not self.vm.simulationIsRunning:
            return

        # mdiWindow = self.vm.addPythonSteeringPanel()

        mdiWindow = self.addPythonSteeringPanel(obj_data)
        self.windowList.append(mdiWindow)
        # self.windowList.append(None)
        _mutex.unlock()

        # newWindow = PlotFrameWidget(self.vm, **obj)
        #
        # newWindow.show()
        #
        # mdiPlotWindow = self.vm.addSubWindow(newWindow)
        #
        # mdiPlotWindow.setWindowTitle(obj['title'])
        #
        # suggested_win_pos = self.vm.suggested_window_position()
        #
        # if suggested_win_pos.x() != -1 and suggested_win_pos.y() != -1:
        #     mdiPlotWindow.move(suggested_win_pos)
        #
        # self.vm.lastActiveRealWindow = mdiPlotWindow
        #
        # print 'mdiPlotWindow=', mdiPlotWindow
        # print 'newWindow=', newWindow
        # newWindow.show()
        #
        # plotWindowInterface = PlotWindowInterface(newWindow)
        # self.plotWindowList.append(plotWindowInterface)  # store plot window interface in the window list

        # self.plotWindowMutex.unlock()

