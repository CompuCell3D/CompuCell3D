# from PlotWindowInterface import PlotWindowInterface
# -*- coding: utf-8 -*-
from .PlotWindowInterface import PlotWindowInterface
from cc3d.player5.Graphics.GraphicsWindowData import GraphicsWindowData
from PyQt5 import QtCore
from cc3d.player5.Graphics.PlotFrameWidget import PlotFrameWidget
# from . import PlotManagerSetup
import cc3d.player5.Configuration as Configuration
from cc3d.core.enums import *


class PlotManager(QtCore.QObject):

    newPlotWindowSignal = QtCore.pyqtSignal(QtCore.QMutex, object)

    def __init__(self, _viewManager=None, _plotSupportFlag=False):
        QtCore.QObject.__init__(self, None)
        self.vm = _viewManager
        self.plotsSupported = _plotSupportFlag
        self.plotWindowList = []
        self.plotWindowMutex = QtCore.QMutex()
        self.signalsInitialized = False

    # def getPlotWindow(self):
    #     if self.plotsSupported:
    #         return PlotWindow()
    #     else:
    #         return PlotWindowBase()

    def reset(self):
        self.plotWindowList = []

    def initSignalAndSlots(self):
        # since initSignalAndSlots can be called in SimTabView multiple times (after each simulation restart) we have to ensure that signals are connected only once
        # otherwise there will be an avalanche of signals - each signal for each additional simulation run this will cause lots of extra windows to pop up

        if not self.signalsInitialized:
            self.newPlotWindowSignal.connect(self.processRequestForNewPlotWindow)
            self.signalsInitialized = True
            # self.connect(self,SIGNAL("newPlotWindow(QtCore.QMutex)"),self.processRequestForNewPlotWindow)

    def restore_plots_layout(self):
        ''' This function restores plot layout - it is called from CompuCellSetup.py inside mainLoopNewPlayer function
        :return: None
        '''

        windows_layout_dict = Configuration.getSetting('WindowsLayout')

        if not windows_layout_dict:
            return

        for winId, win in self.vm.win_inventory.getWindowsItems(PLOT_WINDOW_LABEL):
            plot_frame_widget = win.widget()

            plot_interface = plot_frame_widget.plotInterface()  # plot_frame_widget.plotInterface is a weakref

            if not plot_interface:  # if weakref to plot_interface is None we ignore such window
                continue

            if str(plot_interface.title) in list(windows_layout_dict.keys()):
                window_data_dict = windows_layout_dict[str(plot_interface.title)]



                gwd = GraphicsWindowData()
                gwd.fromDict(window_data_dict)

                if gwd.winType != 'plot':
                    return
                win.resize(gwd.winSize)
                win.move(gwd.winPosition)
                win.setWindowTitle(plot_interface.title)

    def getNewPlotWindow(self, obj=None):


        if obj is None:
            message = "You are most likely using old syntax for scientific plots. When adding new plot window please use " \
                      "the following updated syntax:" \
                      "self.pW = self.addNewPlotWindow" \
                      "(_title='Average Volume And Surface',_xAxisTitle='MonteCarlo Step (MCS)'," \
                      "_yAxisTitle='Variables', _xScaleType='linear',_yScaleType='linear')"

            raise RuntimeError(message)

        self.plotWindowMutex.lock()

        self.newPlotWindowSignal.emit(self.plotWindowMutex, obj)
        # processRequestForNewPlotWindow will be called and it will unlock drawMutex but before it will finish runnning (i.e. before the new window is actually added)we must make sure that getNewPlotwindow does not return
        self.plotWindowMutex.lock()
        self.plotWindowMutex.unlock()
        return self.plotWindowList[-1]  # returning recently added window

    def restoreSingleWindow(self, plotWindowInterface):
        '''
        Restores size and position of a single, just-added plot window
        :param plotWindowInterface: an insance of PlotWindowInterface - can be fetchet from PlotFrameWidget using PlotFrameWidgetInstance.plotInterface
        :return: None
        '''

        windows_layout_dict = Configuration.getSetting('WindowsLayout')
        # print 'windowsLayoutDict=', windowsLayoutDict

        if not windows_layout_dict:
            return

        if str(plotWindowInterface.title) in list(windows_layout_dict.keys()):
            window_data_dict = windows_layout_dict[str(plotWindowInterface.title)]

            gwd = GraphicsWindowData()
            gwd.fromDict(window_data_dict)

            if gwd.winType != 'plot':
                return

            plot_window = self.vm.lastActiveRealWindow
            plot_window.resize(gwd.winSize)
            plot_window.move(gwd.winPosition)
            plot_window.setWindowTitle(plotWindowInterface.title)

    def getPlotWindowsLayoutDict(self):
        windowsLayout = {}

        for winId, win in self.vm.win_inventory.getWindowsItems(PLOT_WINDOW_LABEL):
            plotFrameWidget = win.widget()
            plotInterface = plotFrameWidget.plotInterface()  # getting weakref
            if not plotInterface:
                continue

            gwd = GraphicsWindowData()
            gwd.sceneName = plotInterface.title
            gwd.winType = 'plot'
            plotWindow = plotInterface.plotWindow
            mdiPlotWindow = win
            # mdiPlotWindow = self.vm.findMDISubWindowForWidget(plotWindow)
            print('plotWindow=', plotWindow)
            print('mdiPlotWindow=', mdiPlotWindow)
            gwd.winSize = mdiPlotWindow.size()
            gwd.winPosition = mdiPlotWindow.pos()

            windowsLayout[gwd.sceneName] = gwd.toDict()

        return windowsLayout


    def processRequestForNewPlotWindow(self, _mutex, obj):
        print('obj=', obj)
        #        print MODULENAME,"processRequestForNewPlotWindow(): GOT HERE mutex=",_mutex
        if not self.plotsSupported:
            return PlotWindowInterfaceBase(None)  # dummy PlotwindowInterface


        if not self.vm.simulationIsRunning:
            return

        newWindow = PlotFrameWidget(self.vm, **obj)

        newWindow.show()

        mdiPlotWindow = self.vm.addSubWindow(newWindow)

        mdiPlotWindow.setWindowTitle(obj['title'])

        suggested_win_pos = self.vm.suggested_window_position()

        if suggested_win_pos.x() != -1 and suggested_win_pos.y() != -1:
            mdiPlotWindow.move(suggested_win_pos)

        self.vm.lastActiveRealWindow = mdiPlotWindow

        print('mdiPlotWindow=', mdiPlotWindow)
        print('newWindow=', newWindow)
        newWindow.show()

        plotWindowInterface = PlotWindowInterface(newWindow)
        self.plotWindowList.append(plotWindowInterface)  # store plot window interface in the window list

        # self.restoreSingleWindow(plotWindowInterface)

        self.plotWindowMutex.unlock()

