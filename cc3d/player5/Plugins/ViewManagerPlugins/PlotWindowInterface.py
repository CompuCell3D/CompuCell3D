# -*- coding: utf-8 -*-
import weakref
from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import numpy as np

import warnings
from deprecated import deprecated

try:
    import webcolors as wc
except ImportError:
    warnings.warn('Could not find webcolors. Run "pip install webcolors" to fix this', RuntimeWarning)

import pyqtgraph as pg
# pg.setConfigOption('background', 'w')
import pyqtgraph.exporters

from . import PlotManagerSetup
import os
from cc3d.core.enums import *

MODULENAME = '---- PlotManager.py: '

PLOT_TYPE_POSITION = 3
(XYPLOT, HISTOGRAM, BARPLOT) = list(range(0, 3))
MAX_FIELD_LEGTH = 25


# todo
# Notice histogram and Bar Plot implementations need more work.
# They are functional but have a bit strange syntax and for Bar Plot we can only plot one series per plot

# class PlotWindowInterface(PlotManagerSetup.PlotWindowInterfaceBase,QtCore.QObject):
class PlotWindowInterface(QtCore.QObject):
    # IMPORTANT: in QT4 it turns out that using QString in signals is essential
    # to get correct behavior.
    # for some reason using char * in signal may malfunction e.g. wrong string is sent to slot.

    showPlotSignal = QtCore.pyqtSignal(str, QtCore.QMutex)
    showAllPlotsSignal = QtCore.pyqtSignal(QtCore.QMutex)

    # savePlotAsPNG has to emit signal with locking mutex to work correctly
    savePlotAsPNGSignal = QtCore.pyqtSignal(str, int, int, QtCore.QMutex)

    setTitleSignal = QtCore.pyqtSignal(str)
    setTitleSizeSignal = QtCore.pyqtSignal(int)
    setPlotBackgroundColorSignal = QtCore.pyqtSignal(str)

    addPlotSignal = QtCore.pyqtSignal(object, QtCore.QMutex)

    def __init__(self, _plotWindow=None):
        # PlotManagerSetup.PlotWindowInterfaceBase.__init__(self,_plotWindow)
        QtCore.QObject.__init__(self, None)
        if _plotWindow:
            self.plotWindow = _plotWindow

            self.plotWindow.plotInterface = weakref.ref(self)
            self.pW = self.plotWindow.plotWidget

        self.plot_params = self.plotWindow.getPlotParams()

        self.plotData = {}
        self.plot_data = self.plotData
        self.plotHistData = {}
        self.plotDrawingObjects = {}
        self.initSignalsAndSlots()
        self.plotWindowInterfaceMutex = QtCore.QMutex()
        # this is the index of the flag tha is used to signal wheather the data has been modified or not
        self.dirtyFlagIndex = 2
        self.autoLegendFlag = False
        self.legendSetFlag = False
        # self.legendPosition = Qwt.QwtPlot.BottomLegend
        # todo
        self.legendPosition = None

        self.legend_added = False

        self.barplot = None

        self.eraseAllFlag = False
        self.logScaleFlag = False
        try:
            self.title = self.plot_params['title']
        except:
            self.title = 'GENERIC PLOT'

    def initSignalsAndSlots(self):
        self.showAllPlotsSignal.connect(self.__show_all_plots_handler)
        self.savePlotAsPNGSignal.connect(self.__save_plot_as_png_handler)
        self.setTitleSignal.connect(self.set_title_handler)
        self.setPlotBackgroundColorSignal.connect(self.set_plot_background_color_handler)
        self.addPlotSignal.connect(self.add_plot_handler)

    def clear(self):
        # self.pW.clear()
        self.pW.detachItems()

    def replot(self):
        self.pW.replot()

    # @deprecated(version='4.0.0', reason="You should use : set_title_handler")
    # def setTitleHandler(self, _title):
    #     return self.set_title_handler(title=_title)

    def set_title_handler(self, title):
        self.title = str(title)
        self.pW.setTitle(title)

    @deprecated(version='4.0.0', reason="You should use : set_title")
    def setTitle(self, _title):
        return self.set_title(title=_title)

    def set_title(self, title):
        self.title = str(title)
        self.setTitleSignal.emit(title)
        # self.pW.setTitle(_title)

    @deprecated(version='4.0.0', reason="You should use : set_title_size")
    def setTitleSize(self, _size):
        return self.set_title_size(size=_size)

    def set_title_size(self, size):
        self.setTitleSizeSignal.emit(size)

    @deprecated(version='4.0.0', reason="You should use : set_title_color")
    def setTitleColor(self, _colorName):
        return self.set_title_color(color_name=_colorName)

    def set_title_color(self, color_name):
        try:
            title = self.pW.title()
            title.setColor(QColor(color_name))
            self.pW.set_title(title)
        except:
            raise RuntimeError('setTitleColor function is not supported in Player 5')

    def set_plot_background_color_handler(self, color_name):
        print('_colorName=', color_name)
        # self.pW.setCanvasBackground(QColor(_colorName))

    # @deprecated(version='4.0.0', reason="You should use : set_plot_background_color")
    # def setPlotBackgroundColor(self, _colorName):
    #     return self.set_plot_background_color(color_name=_colorName)

    def set_plot_background_color(self, color_name):
        self.setPlotBackgroundColorSignal.emit(color_name)
        self.pW.getViewBox().setBackgroundColor((255, 255, 255, 255))

        # self.pW.setCanvasBackground(QColor(_colorName))

    @deprecated(version='4.0.0', reason="You should use : set_data_default")
    def setDataDefault(self, plot_obj, x, y):
        return self.set_data_default(plot_obj=plot_obj, x=x, y=y)

    def set_data_default(self, plot_obj, x, y):
        plot_obj.setData(x, y)

    @deprecated(version='4.0.0', reason="You should use : set_data_bar_graph_item")
    def setDataBarGraphItem(self, plot_obj, x, y):
        return self.set_data_bar_graph_item(plot_obj=plot_obj, x=x, y=y)

    def set_data_bar_graph_item(self, plot_obj, x, y):
        plot_obj.setOpts(x=x, height=y)

    @deprecated(version='4.0.0', reason="You should use : add_plot")
    def addPlot(self, _plotName, _style="Lines", _color='white', _size=3, _alpha=255):
        return self.add_plot(plot_name=_plotName, style=_style, color=_color, size=_size, alpha=_alpha)

    def add_plot(self, plot_name, style="Lines", color='white', size=3, alpha=255):
        plot_param_dict = {
            '_plotName': plot_name,
            '_style': style,
            '_color': color,
            '_size': size,
            '_alpha': alpha
        }

        self.plotWindowInterfaceMutex.lock()
        self.addPlotSignal.emit(plot_param_dict, self.plotWindowInterfaceMutex)

        self.plotWindowInterfaceMutex.lock()
        self.plotWindowInterfaceMutex.unlock()

    # @deprecated(version='4.0.0', reason="You should use : add_plot_handler")
    # def addPlotHandler(self, plot_param_dict):
    #     return self.add_plot_handler(plot_param_dict=plot_param_dict)

    def add_plot_handler(self, plot_param_dict):

        plot_name = plot_param_dict['_plotName']
        style = plot_param_dict['_style']
        color = plot_param_dict['_color']

        background_color = self.pW.backgroundBrush().color()

        # print 'dir(self.pW)=',dir(self.pW)

        size = plot_param_dict['_size']
        alpha = plot_param_dict['_alpha']

        add_legend = False
        try:
            add_legend = self.plot_params['legend']
        except KeyError:
            pass

        if add_legend and not self.legend_added:
            self.pW.addLegend()
            self.legend_added = True

        alpha = abs(int(alpha)) % 256

        yd, xd = np.array([], dtype=np.float), np.array([], dtype=np.float)

        set_data_fcn = self.set_data_default

        color = wc.name_to_rgb(color) + (alpha,)
        if style.lower() == 'dots':
            plot_obj = self.pW.plot(y=yd, x=xd, pen=background_color, symbolBrush=color, symbolSize=size,
                                   name=plot_name)

        elif style.lower() == 'lines':
            pen = pg.mkPen(color=color, width=size)
            plot_obj = self.pW.plot(y=yd, x=xd, pen=pen, name=plot_name)

        elif style.lower() == 'steps':
            xd, yd = np.array([0, .00001], dtype=np.float), np.array([1], dtype=np.float)
            # pen = pg.mkPen(color=color, width=size)
            # plotObj = self.pW.plot(y=yd, x=xd, pen=pen,stepMode=True)

            plot_obj = pg.PlotCurveItem(xd, yd, stepMode=True, fillLevel=0, brush=color, name=plot_name)

            self.pW.addItem(plot_obj)
            # plt1.addItem(curve)

        elif style.lower() == 'bars':

            plot_obj = pg.BarGraphItem(x=xd, height=yd, width=size, brush=color, name=plot_name)
            set_data_fcn = self.set_data_bar_graph_item
            self.pW.addItem(plot_obj)

        else:  # dots is the default
            plot_obj = self.pW.plot(y=yd, x=xd, pen=background_color, symbolBrush=color, symbolSize=size,
                                   name=plot_name)

        self.plotData[plot_name] = [xd, yd, False, XYPLOT, False]
        self.plotDrawingObjects[plot_name] = {'curve': plot_obj,
                                              'LineWidth': size,
                                              'LineColor': color,
                                              'Style': style,
                                              'SetData': set_data_fcn}

        self.plotWindowInterfaceMutex.unlock()

    @deprecated(version='4.0.0', reason="You should use : erase_all_data")
    def eraseAllData(self):
        return self.erase_all_data()

    def erase_all_data(self):
        """
        erases all pot data - effectively this clears plots
        :return:
        """
        self.clean_all_containers()

        for name, data in self.plotData.items():
            data[self.dirtyFlagIndex] = True

        self.eraseAllFlag = True

    def clean_all_containers(self):
        """
        Cleans all data containers that store plot data. Helper function
        :return:
        """

        for name, data in self.plotData.items():
            # todo implement special handling for "steps" or switch to bars instead of steps
            if data[0].shape[0] == data[1].shape[0] + 1:  # steps
                data[0], data[1] = np.array([0, .00001], dtype=np.float), np.array([0], dtype=np.float)
                data[4] = False
            else:
                data[0], data[1] = np.array([], dtype=np.float), np.array([], dtype=np.float)
            data[self.dirtyFlagIndex] = True


    def erase_data(self, plot_name:str):
        """
        Erases data for a particular plot
        :param plot_name: plot name for which data will be erased
        :return:
        """
        plot_type = self.plotData[plot_name][PLOT_TYPE_POSITION]
        self.plotData[plot_name] = [np.array([], dtype=np.float), np.array([], dtype=np.float), False, plot_type]

    @deprecated(version='4.0.0', reason="You should use : add_data_series")
    def addDataSeries(self, _plotName, _x_vec, _y_vec):
        return self.add_data_series(plot_name=_plotName, x_vec=_x_vec, y_vec=_y_vec)

    def add_data_series(self, plot_name, x_vec, y_vec):

        if not isinstance(x_vec, (list, tuple, np.ndarray)):
            raise RuntimeError('addDataSeries: _x_vec has to be a list, tuple or 1D numpe array')

        if not isinstance(y_vec, (list, tuple, np.ndarray)):
            raise RuntimeError('addDataSeries: _y_vec has to be a list, tuple or 1D numpe array')

        if len(x_vec) != len(y_vec):
            raise RuntimeError('addDataSeries: _x_vec and _y_vec have to be of the same length')

        for x, y in zip(x_vec, y_vec):
            self.add_data_point(plot_name=plot_name, x=x, y=y)

    @deprecated(version='4.0.0', reason="You should use : add_data_point")
    def addDataPoint(self, _plotName, _x, _y):
        return self.add_data_point(plot_name=_plotName, x=_x, y=_y)

    def add_data_point(self, plot_name, x, y):

        # print('add_data_point: self.plotData=', self.plotData)

        if plot_name not in list(self.plotData.keys()):
            return

        if self.eraseAllFlag:
            self.clean_all_containers()
            self.eraseAllFlag = False

        currentLength = len(self.plotData[plot_name][0])

        x_vec = self.plotData[plot_name][0]
        y_vec = self.plotData[plot_name][1]

        plot_obj = self.plotDrawingObjects[plot_name]
        style = plot_obj['Style']
        if style.lower() == 'steps':
            # processing first first point of the histogram
            if not self.plotData[plot_name][4]:

                x_vec = np.array([x, x + 1], dtype=np.float)
                y_vec = np.array([y], dtype=np.float)

                self.plotData[plot_name][0] = x_vec
                self.plotData[plot_name][1] = y_vec
                self.plotData[plot_name][4] = True
            else:
                self.plotData[plot_name][0] = np.append(self.plotData[plot_name][0], [x])
                self.plotData[plot_name][1] = np.append(self.plotData[plot_name][1], [y])
                self.plotData[plot_name][0][-2] = x
                self.plotData[plot_name][0][-1] = x + 1

        else:
            self.plotData[plot_name][0] = np.append(self.plotData[plot_name][0], [x])
            self.plotData[plot_name][1] = np.append(self.plotData[plot_name][1], [y])

        self.plotData[plot_name][self.dirtyFlagIndex] = True

        # print('exit add_data_point: self.plotData=', self.plotData)

    def get_drawing_objects_settings(self, plot_name:str):
        """
        returns settings for plot
        :param plot_name: plot name
        :return:
        """
        if plot_name in list(self.plotDrawingObjects.keys()):
            return self.plotDrawingObjects[plot_name]
        else:
            return None

    @deprecated(version='4.0.0', reason="You should use : save_plot_as_png")
    def savePlotAsPNG(self, _fileName, _sizeX=400, _sizeY=400):
        return self.save_plot_as_png(file_name=_fileName, size_x=_sizeX, size_y=_sizeY)

    def save_plot_as_png(self, file_name, size_x=400, size_y=400):
        """
        writes plot as png to the drive.Current implementation cannot resize image size but this may change
        in the future therefore we are keeping 'size_x', 'size_y' variables

        :param file_name: {str} file name
        :param size_x: {int} image x-size -  currently not used
        :param size_y: {int} image y-size -  currently not used
        :return:
        """

        self.plotWindowInterfaceMutex.lock()
        self.savePlotAsPNGSignal.emit(str(file_name), size_x, size_y, self.plotWindowInterfaceMutex)

        self.plotWindowInterfaceMutex.lock()
        self.plotWindowInterfaceMutex.unlock()

    def __save_plot_as_png_handler(self, file_name, size_x, size_y, mutex):
        """
        Hendler - writes plot as png to the drive.Current implementation cannot resize image size but this may change
        in the future therefore we are keeping 'size_x', 'size_y' variables

        :param file_name: {str} file name
        :param size_x: {int} image x-size -  currently not used
        :param size_y: {int} image y-size -  currently not used
        :param mutex: {QMutex}
        :return: None
        """

        warnings.warn('Player 5 does not allow scaling of plot screenshots. If this feature is required,'
                      'it is best to save plot data and render it '
                      'separately in a full-featured plotting package such as Matpotlib or pyqtgraph. '
                      'CompuCell3D provides only basic plotting capabilities', RuntimeWarning)

        file_name = str(file_name)
        pixmap = QPixmap(self.pW.size())
        painter = QPainter()
        painter.begin(pixmap)
        self.pW.render(painter, self.pW.sceneRect())
        pixmap.save(file_name)
        painter.end()

        mutex.unlock()

    def write_out_header(self, fp, plot_name, output_format=LEGACY_FORMAT):
        """
        Writes header for plot data seres. Used as a convenience function by the function that writes
        plot data to the disk
        :param fp: file pointer
        :param plot_name: plot name
        :param output_format: outpout format
        :return:
        """
        if output_format == LEGACY_FORMAT:

            fp.write(plot_name + '\n')
            return 0  # field width

        elif output_format == CSV_FORMAT:

            plot_name = plot_name.replace(' ', '_')

            field_size = len(plot_name) + 2  # +2 is for _x or _y
            if MAX_FIELD_LEGTH > field_size:
                field_size = MAX_FIELD_LEGTH

            fmt = ''
            fmt += '{0:>' + str(field_size) + '},'
            fmt += '{1:>' + str(field_size) + '}\n'

            fp.write(fmt.format(plot_name + '_x', plot_name + '_y'))

            return field_size

        else:
            raise LookupError(
                MODULENAME + " writeOutHeader :" + "Requested output format: " + output_format + " does not exist")

    @deprecated(version='4.0.0', reason="You should use : save_plot_as_data")
    def savePlotAsData(self, _fileName, _outputFormat=LEGACY_FORMAT):
        return self.save_plot_as_data(file_name=_fileName, output_format=_outputFormat)

    def save_plot_as_data(self, file_name, output_format=LEGACY_FORMAT):
        """
        Writes plots' data as a data file
        :param file_name: {str} file name
        :param output_format: output format
        :return:
        """

        outfile = file_name
        fpout = open(outfile, "w")

        for plot_name, plot_data in self.plotData.items():
            fieldSize = self.write_out_header(fp=fpout, plot_name=plot_name, output_format=output_format)
            xvals = plot_data[0]
            yvals = plot_data[1]
            if output_format == LEGACY_FORMAT:
                if plot_data[PLOT_TYPE_POSITION] == XYPLOT or plot_data[PLOT_TYPE_POSITION] == BARPLOT:
                    for jdx in range(len(xvals)):

                        # we need to intercept Index Error because steps data appends extra data point in x array
                        try:
                            xy_str = "%f  %f\n" % (xvals[jdx], yvals[jdx])
                            fpout.write(xy_str)
                        except IndexError:
                            pass

                elif plot_data[PLOT_TYPE_POSITION] == HISTOGRAM:
                    for jdx in range(len(xvals) - 1):

                        # we need to intercept Index Error because steps data appends extra data point in x array
                        try:
                            xy_str = "%f  %f\n" % (xvals[jdx], yvals[jdx])
                            fpout.write(xy_str)
                        except IndexError:
                            pass

            elif output_format == CSV_FORMAT:
                fmt = ''
                fmt += '{0:>' + str(fieldSize) + '},'
                fmt += '{1:>' + str(fieldSize) + '}\n'

                if plot_data[PLOT_TYPE_POSITION] == XYPLOT or plot_data[PLOT_TYPE_POSITION] == BARPLOT:
                    for jdx in range(len(xvals)):

                        # we need to intercept Index Error because steps data appends extra data point in x array
                        try:
                            xy_str = fmt.format(xvals[jdx], yvals[jdx])
                            fpout.write(xy_str)
                        except IndexError:
                            pass
                elif plot_data[PLOT_TYPE_POSITION] == HISTOGRAM:
                    for jdx in range(len(xvals) - 1):

                        # we need to intercept Index Error because steps data appends extra data point in x array
                        try:
                            xy_str = fmt.format(xvals[jdx], yvals[jdx])
                            fpout.write(xy_str)
                        except IndexError:
                            pass

            else:
                raise LookupError(
                    MODULENAME + " savePlotAsData :" + "Requested output format: " + outputFormat + " does not exist")
            fpout.write('\n')  # separating data series by a line

        fpout.close()

    @deprecated(version='4.0.0', reason="You should use : show_all_plots")
    def showAllPlots(self):
        return self.show_all_plots()

    def show_all_plots(self):
        """
        Updates aall plots with current data. Effectively shows plots onthe screen
        :return:
        """
        self.plotWindowInterfaceMutex.lock()
        self.showAllPlotsSignal.emit(self.plotWindowInterfaceMutex)

        self.plotWindowInterfaceMutex.lock()
        self.plotWindowInterfaceMutex.unlock()

    def __show_all_plots_handler(self, _mutex=None):

        for plotName in list(self.plotData.keys()):
            if self.plotData[plotName][self.dirtyFlagIndex]:
                if plotName in list(self.plotDrawingObjects.keys()):
                    x_vec = self.plotData[plotName][0]
                    y_vec = self.plotData[plotName][1]

                    plotObj = self.plotDrawingObjects[plotName]['curve']
                    # print 'plotName=',plotName
                    # print 'x_vec=',x_vec
                    # print 'y_vec=', y_vec
                    setData_fcn = self.plotDrawingObjects[plotName]['SetData']
                    setData_fcn(plotObj, x_vec, y_vec)

                    # todo OK
                    # plotObj.setData(x_vec,y_vec)
                    self.plotData[plotName][self.dirtyFlagIndex] = False

        _mutex.unlock()

    @deprecated(version='4.0.0', reason="You should use : add_histogram")
    def addHistogram(self, plot_name, value_array, number_of_bins):
        return self.add_histogram(plot_name=plot_name, value_array=value_array, number_of_bins=number_of_bins)

    def add_histogram(self, plot_name, value_array, number_of_bins):
        """
        Creates a histogram out of "value_array" and adds it to a histogram plotplot
        :param plot_name:
        :param value_array:
        :param number_of_bins:
        :return:
        """

        (values, intervals) = np.histogram(value_array, bins=number_of_bins)

        self.plotData[plot_name][0] = intervals
        self.plotData[plot_name][1] = values

        self.plotData[plot_name][self.dirtyFlagIndex] = True

    @deprecated(version='4.0.0', reason="You should use : add_histogram_plot")
    def addHistogramPlot(self, _plotName, _color='blue', _alpha=255):
        return self.add_histogram_plot(plot_name=_plotName, color=_color, alpha=_alpha)

    def add_histogram_plot(self, plot_name, color='blue', alpha=255):
        """
        Adds empty historgram plot
        :param plot_name:
        :param color:
        :param alpha:
        :return:
        """
        self.add_plot(plot_name=plot_name, style='Steps', color=color, size=1.0, alpha=alpha)

