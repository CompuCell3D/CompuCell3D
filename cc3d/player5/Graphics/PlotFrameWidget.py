import warnings

try:
    import webcolors as wc
except ImportError:
    warnings.warn('Could not find webcolors. Run "pip install webcolors" to fix this', RuntimeWarning)

import sys
from PyQt5 import QtCore, QtGui, QtOpenGL
import pyqtgraph as pg


class PlotFrameWidget(QtGui.QFrame):
    def __init__(self, parent=None, **kwds):
        QtGui.QFrame.__init__(self, parent)

        self.plot_params = kwds
        print('kwds=', kwds)

        # self.plotWidget=CartesianPlot()
        # self.plotWidget = pg.PlotWidget(background='w')
        self.plotWidget = pg.PlotWidget()

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


        # self.plotWidget = pg.GraphicsView()
        self.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding))

        self.plotInterface = None

        self.parentWidget = parent
        layout = QtGui.QBoxLayout(QtGui.QBoxLayout.TopToBottom)
        layout.addWidget(self.plotWidget)

        # self.setWindowTitle(kwds['title']) # setting title bar on the window
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
        # self.resize(600, 600)
        self.setMinimumSize(100, 100)  # needs to be defined to resize smaller than 400x400


    def resizePlot(self, x, y):
        self.plotWidget.sizeHint = QtCore.QSize(x, y)
        self.plotWidget.resize(self.plotWidget.sizeHint)
        self.resize(self.plotWidget.sizeHint)


    # # note that if you close widget using X button this slot is not called
    # # we need to reimplement closeEvent
    # # def close(self):

    def getPlotParams(self):
        """
        Fetches a dictionary of parameters describing plot
        @return: {dict}
        """
        return self.plot_params

    def closeEvent(self, ev):
        pass
        # self.parentWidget.closeActiveSubWindowSlot()

