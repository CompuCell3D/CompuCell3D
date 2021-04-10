from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from weakref import ref

from cc3d.player5.steering.SteeringPanelView import SteeringPanelView
from cc3d.player5.steering.SteeringPanelModel import SteeringPanelModel
from cc3d.player5.steering.SteeringEditorDelegate import SteeringEditorDelegate


class WidgetManager(QtCore.QObject):
    """
    This class will manage creation and destruction of widgets that the user will request from Python Steppables
    """

    newWidgetSignal = QtCore.pyqtSignal(QtCore.QMutex, object, object)

    def __init__(self, _viewManager=None, _plotSupportFlag=False):
        QtCore.QObject.__init__(self, None)
        self.vm = _viewManager
        self.windowMutex = QtCore.QMutex()
        self.signalsInitialized = False
        self.windowList = []

    @property
    def vm(self):
        try:
            o = self._vm()
        except TypeError:
            o = self._vm
        return o

    @vm.setter
    def vm(self, _i):
        try:
            self._vm = ref(_i)
        except TypeError:
            self._vm = _i

    def initSignalAndSlots(self):
        # since initSignalAndSlots can be called in SimTabView multiple times
        # (after each simulation restart) we have to ensure that signals are connected only once
        # otherwise there will be an avalanche of signals - each signal for each additional
        # simulation run this will cause lots of extra windows to pop up

        if not self.signalsInitialized:
            self.newWidgetSignal.connect(self.processRequestForNewWidget)
            self.signalsInitialized = True
            # self.connect(self,SIGNAL("newPlotWindow(QtCore.QMutex)"),self.processRequestForNewPlotWindow)

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

        print('THIS IS ADD STEERING PANEL')
        # from steering.SteeringPanelView import SteeringPanelView
        # from steering.SteeringPanelModel import SteeringPanelModel
        # from steering.SteeringEditorDelegate import SteeringEditorDelegate

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
