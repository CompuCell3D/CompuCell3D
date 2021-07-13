from .PlotWindowInterface import PlotWindowInterface
from cc3d.player5.Graphics.GraphicsWindowData import GraphicsWindowData
from PyQt5 import QtCore
from cc3d.player5.Graphics.PlotFrameWidget import PlotFrameWidget
import cc3d.player5.Configuration as Configuration
from cc3d.core.enums import *
from cc3d.player5.Plugins.ViewManagerPlugins.PlotManagerBase import PlotManagerBase


class PlotManager(QtCore.QObject, PlotManagerBase):

    newPlotWindowSignal = QtCore.pyqtSignal(QtCore.QMutex, object)

    def __init__(self, view_manager=None, plot_support_flag=False):
        QtCore.QObject.__init__(self, None)
        PlotManagerBase.__init__(self, view_manager, plot_support_flag)

        self.plotWindowList = []
        self.plotWindowMutex = QtCore.QMutex()
        self.signalsInitialized = False

    def reset(self):
        self.plotWindowList = []

    def init_signal_and_slots(self):
        # since initSignalAndSlots can be called in SimTabView multiple times
        # (after each simulation restart) we have to ensure that signals are connected only once
        # otherwise there will be an avalanche of signals - each signal for each additional
        # simulation run this will cause lots of extra windows to pop up

        if not self.signalsInitialized:
            self.newPlotWindowSignal.connect(self.process_request_for_new_plot_window)
            self.signalsInitialized = True

    def restore_plots_layout(self):
        """
        This function restores plot layout - it is called from CompuCellSetup.py inside mainLoopNewPlayer function
        :return:
        """

        windows_layout_dict = Configuration.getSetting('WindowsLayout')

        if not windows_layout_dict:
            return

        for winId, win in self.vm.win_inventory.getWindowsItems(PLOT_WINDOW_LABEL):
            plot_frame_widget = win.widget()

            # plot_frame_widget.plotInterface is a weakref
            plot_interface = plot_frame_widget.plotInterface()

            # if weakref to plot_interface is None we ignore such window
            if not plot_interface:
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

    def get_new_plot_window(self, obj=None):
        """
        Returns recently added plot window
        :param obj:
        :return:
        """

        if obj is None:
            message = "You are most likely using old syntax for scientific plots. " \
                      "When adding new plot window please use " \
                      "the following updated syntax:" \
                      "self.pW = self.addNewPlotWindow" \
                      "(_title='Average Volume And Surface',_xAxisTitle='MonteCarlo Step (MCS)'," \
                      "_yAxisTitle='Variables', _xScaleType='linear',_yScaleType='linear')"

            raise RuntimeError(message)

        self.plotWindowMutex.lock()

        self.newPlotWindowSignal.emit(self.plotWindowMutex, obj)
        # processRequestForNewPlotWindow will be called and it will
        # unlock drawMutex but before it will finish running
        # (i.e. before the new window is actually added)we must make sure that getNewPlotwindow does not return
        self.plotWindowMutex.lock()
        self.plotWindowMutex.unlock()

        # returning recently added window
        return self.plotWindowList[-1]

    def get_plot_windows_layout_dict(self):
        windows_layout = {}

        for winId, win in self.vm.win_inventory.getWindowsItems(PLOT_WINDOW_LABEL):
            plot_frame_widget = win.widget()
            # getting weakref
            plot_interface = plot_frame_widget.plotInterface()
            if not plot_interface:
                continue

            gwd = GraphicsWindowData()
            gwd.sceneName = plot_interface.title
            gwd.winType = 'plot'
            plot_window = plot_interface.plotWindow
            mdi_plot_window = win
            gwd.winSize = mdi_plot_window.size()
            gwd.winPosition = mdi_plot_window.pos()

            windows_layout[gwd.sceneName] = gwd.toDict()

        return windows_layout

    def process_request_for_new_plot_window(self, _mutex, obj):

        if not self.vm.simulationIsRunning:
            return

        new_window = PlotFrameWidget(self.vm, **obj)

        new_window.show()

        mdi_plot_window = self.vm.addSubWindow(new_window)

        mdi_plot_window.setWindowTitle(obj['title'])

        suggested_win_pos = self.vm.suggested_window_position()

        if suggested_win_pos.x() != -1 and suggested_win_pos.y() != -1:
            mdi_plot_window.move(suggested_win_pos)

        self.vm.lastActiveRealWindow = mdi_plot_window

        new_window.show()

        plot_window_interface = PlotWindowInterface(new_window)
        # store plot window interface in the window list
        self.plotWindowList.append(plot_window_interface)

        self.plotWindowMutex.unlock()
