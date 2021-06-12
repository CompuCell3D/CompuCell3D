import warnings
from weakref import ref

try:
    import webcolors as wc
except ImportError:
    warnings.warn('Could not find webcolors. Run "pip install webcolors" to fix this', RuntimeWarning)

import sys
from PyQt5 import QtCore, QtGui, QtOpenGL
import pyqtgraph as pg
from pyqtgraph.graphicsItems.PlotItem import PlotItem


class PlotFrameWidget(QtGui.QFrame):
    def __init__(self, parent=None, **kwds):
        QtGui.QFrame.__init__(self, parent)

        self.plot_params = kwds
        self.plotWidget = pg.PlotWidget()

        self.tweak_context_menu(plot_item=self.plotWidget.plotItem)

        try:
            bg_color = kwds['background']
        except LookupError:
            bg_color = None

        if bg_color:
            try:
                bg_color_rgb = wc.name_to_rgb(bg_color)
                self.plotWidget.setBackground(background=bg_color_rgb)
            except ValueError as e:
                print('Could not decode the color %s : Exception : %s'%(bg_color, str(e)), file=sys.stderr)

        self.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding))

        self.plotInterface = None

        self.parentWidget = parent
        layout = QtGui.QBoxLayout(QtGui.QBoxLayout.TopToBottom)
        layout.addWidget(self.plotWidget)

        self.plotWidget.setTitle(kwds['title'])
        self.plotWidget.setLabel(axis='bottom', text=kwds['xAxisTitle'])
        self.plotWidget.setLabel(axis='left', text=kwds['yAxisTitle'])
        x_log_flag, y_log_flag = False, False
        if kwds['xScaleType'].strip().lower() == 'log':
            x_log_flag = True

        if kwds['yScaleType'].strip().lower() == 'log':
            y_log_flag = True

        self.plotWidget.setLogMode(x=x_log_flag, y=y_log_flag)
        if kwds['grid']:
            self.plotWidget.showGrid(x=True, y=True, alpha=1.0)

        self.setLayout(layout)
        # needs to be defined to resize smaller than 400x400
        self.setMinimumSize(100, 100)

    @property
    def parentWidget(self):
        """
        Parent if any, otherwise None
        """
        try:
            o = self._parentWidget()
        except TypeError:
            o = self._parentWidget
        return o

    @parentWidget.setter
    def parentWidget(self, _i):
        try:
            self._parentWidget = ref(_i)
        except TypeError:
            self._parentWidget = _i

    @staticmethod
    def tweak_context_menu(plot_item: PlotItem):
        """
        We are turning off some options for plot's context menus if they are known to cause troubles
        Because we are dealing with various pyqtgraph versions the code will need to consider this
        context menu actions are defined in and are accessible via pyqtgrtaph.graphicsItems.PlotItem.PlotItem
        if you look in the constructor of the PlotItem you will fine lines like:
        c.fftCheck.toggled.connect(self.updateSpectrumMode)
        replace c with
        plot_item.ctrl and then you can turn off actions as you see fit

        :param plot_item:
        :return:
        """
        pg_version_list = pg.__version__.split('.')
        major_ver = int(pg_version_list[0])
        minor_ver = int(pg_version_list[1])

        plot_item.ctrl.fftCheck.setEnabled(False)

        if major_ver <= 0 and minor_ver < 11:
            plot_item.ctrl.fftCheck.setEnabled(False)
            plot_item.ctrl.logXCheck.setEnabled(False)
            plot_item.ctrl.logYCheck.setEnabled(False)
            plot_item.ctrl.downsampleCheck.setEnabled(False)

    def resizePlot(self, x, y):
        self.plotWidget.sizeHint = QtCore.QSize(x, y)
        self.plotWidget.resize(self.plotWidget.sizeHint)
        self.resize(self.plotWidget.sizeHint)

    def getPlotParams(self):
        """
        Fetches a dictionary of parameters describing plot

        :return: {dict}
        """
        return self.plot_params

    def closeEvent(self, ev):
        pass
