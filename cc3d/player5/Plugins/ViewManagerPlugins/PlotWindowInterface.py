# -*- coding: utf-8 -*-
import weakref
from PyQt5 import QtCore
from PyQt5.QtGui import *
import numpy as np
from collections import OrderedDict
from typing import Iterable

import warnings
from deprecated import deprecated

try:
    import webcolors as wc
except ImportError:
    warnings.warn('Could not find webcolors. Run "pip install webcolors" to fix this', RuntimeWarning)

import pyqtgraph as pg
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

        self.plotData = OrderedDict()
        self.plot_data = self.plotData
        self.plotHistData = OrderedDict()
        self.plotDrawingObjects = OrderedDict()
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
        self.right_axis_count = 0

        self.first_plot_item = None
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
        self.pW.detachItems()

    def replot(self):
        self.pW.replot()

    def set_title_handler(self, title):
        self.title = str(title)
        self.pW.setTitle(title)

    @deprecated(version='4.0.0', reason="You should use : set_title")
    def setTitle(self, _title):
        return self.set_title(title=_title)

    def set_title(self, title):
        self.title = str(title)
        self.setTitleSignal.emit(title)

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

    def set_plot_background_color(self, color_name):
        self.setPlotBackgroundColorSignal.emit(color_name)
        self.pW.getViewBox().setBackgroundColor((255, 255, 255, 255))

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

    def add_plot(self, plot_name, style="Lines", color='white', size=3, alpha=255,
                 separate_y_axis=False, y_min=None, y_max=None, y_scale_type=None):
        """
        User's API function - adds a data series plot to the plotting window
        :param plot_name:
        :param style:
        :param color:
        :param size:
        :param alpha:
        :param separate_y_axis:
        :param y_min:
        :param y_max:
        :param y_scale_type:
        :return:
        """

        if not separate_y_axis:
            if y_min is not None or y_max is not None:
                raise RuntimeError('y_min or y_max can only be set if you set separate_y_axis=True '
                                   'for this data series')
            if y_scale_type is not None:
                raise RuntimeError('y_scale_type can only be set if you set separate_y_axis=True '
                                   'for this data series')
        else:
            allowed_y_scaled_types = ['linear', 'log']
            if y_scale_type is not None and y_scale_type.lower() not in allowed_y_scaled_types:
                raise RuntimeError(f'y_scale_type can be only one of {allowed_y_scaled_types}')

        plot_param_dict = {
            '_plotName': plot_name,
            '_style': style,
            '_color': color,
            '_size': size,
            '_alpha': alpha,
            'separate_y_axis': separate_y_axis,
            'y_min': y_min,
            'y_max': y_max,
            'y_scale_type': y_scale_type
        }

        self.plotWindowInterfaceMutex.lock()
        self.addPlotSignal.emit(plot_param_dict, self.plotWindowInterfaceMutex)

        self.plotWindowInterfaceMutex.lock()
        self.plotWindowInterfaceMutex.unlock()

    def add_plot_handler(self, plot_param_dict):

        plot_name = plot_param_dict['_plotName']
        style = plot_param_dict['_style']
        color = plot_param_dict['_color']
        separate_y_axis = plot_param_dict['separate_y_axis']

        background_color = self.pW.backgroundBrush().color()

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

        color = wc.name_to_rgb(color) + (alpha,)
        plot_obj, set_data_fcn = self.construct_plot_item(xd=xd, yd=yd, color=color, size=size, plot_name=plot_name,
                                                          style=style)

        self.plotData[plot_name] = [xd, yd, False, XYPLOT, False]

        set_y_range_allowed = plot_param_dict['y_min'] is not None and plot_param_dict['y_max'] is not None

        self.plotDrawingObjects[plot_name] = {'curve': plot_obj,
                                              'LineWidth': size,
                                              'LineColor': color,
                                              'Style': style,
                                              'SetData': set_data_fcn,
                                              'separate_y_axis': separate_y_axis,
                                              'y_scale_type': plot_param_dict['y_scale_type'],
                                              'y_min': plot_param_dict['y_min'],
                                              'y_max': plot_param_dict['y_max'],
                                              'set_y_range_allowed': set_y_range_allowed

                                              }

        self.add_plot_item_to_the_scene(plot_name=plot_name, plot_specs_dict=self.plotDrawingObjects[plot_name])

        self.plotWindowInterfaceMutex.unlock()

    def construct_plot_item(self, xd: Iterable, yd: Iterable, color: Iterable, size: int, plot_name: str, style: str):
        """
        Constructs plot item
        :param xd: x coordinates of data series to plot
        :param yd: y coordinates of data series to plot
        :param color: tuple - with RGBA point color specification
        :param size: size of the marker
        :param plot_name: data series label
        :param style: style of the plot
        :return: (tuple ) plot item , set_data_fcn
        """
        set_data_fcn = self.set_data_default

        if style.lower() == 'dots':

            plot_obj = pg.ScatterPlotItem(y=yd, x=xd, pen=color, size=size, name=plot_name)

        elif style.lower() == 'lines':
            pen = pg.mkPen(color=color, width=size)
            plot_obj = pg.PlotCurveItem(y=yd, x=xd, pen=pen, name=plot_name)

        elif style.lower() == 'steps':
            xd, yd = np.array([0, .00001], dtype=np.float), np.array([1], dtype=np.float)

            plot_obj = pg.PlotCurveItem(xd, yd, stepMode=True, fillLevel=0, brush=color, name=plot_name)

        elif style.lower() == 'bars':

            plot_obj = pg.BarGraphItem(x=xd, height=yd, width=size, brush=color, name=plot_name)
            set_data_fcn = self.set_data_bar_graph_item

        else:
            # dots is the default
            plot_obj = pg.ScatterPlotItem(y=yd, x=xd, pen=color, size=size, name=plot_name)

        return plot_obj, set_data_fcn

    def add_plot_item_to_the_scene(self, plot_name: str, plot_specs_dict: dict):
        """
        adds plot object to existing plot scene
        :param plot_name:
        :param plot_specs_dict:
        :return:
        """

        psd = plot_specs_dict
        plot_obj = psd['curve']
        axis_color = psd['LineColor']
        separate_y_axis = psd['separate_y_axis']
        y_min = psd['y_min']
        y_max = psd['y_max']
        y_scale_type = psd['y_scale_type']
        set_y_range_allowed = psd['set_y_range_allowed']

        y_log_flag = False
        if y_scale_type is not None and y_scale_type.strip().lower() == 'log':
            y_log_flag = True

        if len(self.plotDrawingObjects) == 1:
            self.first_plot_item = self.pW.plotItem
            plot_specs_dict['plot_item_ref'] = weakref.ref(self.first_plot_item)

            # self.pW.getViewBox().addItem(plot_obj)
            self.first_plot_item.vb.sigResized.connect(self.update_views)
            self.pW.addItem(plot_obj)
            # if firstly added item requests to use separate y axis we will rename left y axis to correspond to this
            # data series
            if separate_y_axis:
                y_axis_item = self.first_plot_item.getAxis('left')
                color = wc.rgb_to_hex(axis_color[:3])
                y_axis_item.setLabel(plot_name, color=color)

                y_axis_item.setLogMode(y_log_flag)

            if set_y_range_allowed:
                # print('this should never happen set_y_range_allowed=', set_y_range_allowed)
                # if user specifies both limits we can set it here
                self.first_plot_item.setYRange(y_min, y_max)
        else:
            if separate_y_axis:
                p1 = self.first_plot_item

                if len(self.plotDrawingObjects) == 2:
                    # adding another data series to the same plot window=
                    self.right_axis_count += 1
                    p2 = pg.ViewBox()
                    p1.showAxis('right')
                    p1.scene().addItem(p2)
                    p1.getAxis('right').linkToView(p2)

                    p1.getAxis('right').setLogMode(y_log_flag)
                    p2.setXLink(p1)

                    line_color = wc.rgb_to_hex(axis_color[:3])
                    p1.getAxis('right').setLabel(plot_name, color=line_color)
                    p2.addItem(plot_obj)
                    # plot_obj.setLogMode(x=False, y=y_log_flag)
                    if set_y_range_allowed:
                        # if user specifies both limits we can set it here
                        p2.setYRange(y_min, y_max)

                    plot_specs_dict['plot_item_ref'] = weakref.ref(p2)

                else:
                    # we need to create a new axis as well
                    self.right_axis_count += 1
                    p2 = pg.ViewBox()
                    ax2 = pg.AxisItem('right')
                    ax2.setLogMode(y_log_flag)
                    # note addItem function in general takes parameters that tell where to add an item in terms of
                    # layout position e.g. l.addItem(a2, row = 2, col = 5,  rowspan=1, colspan=1)
                    # see for example
                    # https://stackoverflow.com/questions/29473757/pyqtgraph-multiple-y-axis-on-left-side
                    p1.layout.addItem(ax2, 2, 1 + self.right_axis_count)
                    p1.scene().addItem(p2)
                    ax2.linkToView(p2)
                    p2.setXLink(p1)
                    line_color = wc.rgb_to_hex(axis_color[:3])
                    ax2.setLabel(plot_name, color=line_color)
                    p2.addItem(plot_obj)
                    if set_y_range_allowed:
                        # if user specifies both limits we can set it here
                        p2.setYRange(y_min, y_max)

                    plot_specs_dict['plot_item_ref'] = weakref.ref(p2)

                # adding item to the legend
                legend = self.first_plot_item.legend
                if legend is not None and self.legend_added:
                    legend.addItem(plot_obj, plot_name)

            else:
                self.pW.addItem(plot_obj)

            self.update_views()

    def update_views(self):
        """
        This function ensures that the view boxes that operate with different axes are linked properly and behave
        properly during widget resizing
        The code is based on MultiplePlotAxes.py demo from pyqtgraph
        :return:
        """
        p1 = self.first_plot_item
        for i, (plot_name, plot_drawing_objects_data_dict) in enumerate(self.plotDrawingObjects.items()):

            if i == 0 or not plot_drawing_objects_data_dict['separate_y_axis']:
                # we link link view box of plots second third etc to the view box of plot one
                continue
            plot_item = plot_drawing_objects_data_dict['curve']
            view_box = plot_item.getViewBox()
            try:
                view_box.setGeometry(p1.vb.sceneBoundingRect())
            except AttributeError:
                print
            view_box.linkedViewChanged(p1.vb, view_box.XAxis)

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

    def erase_data(self, plot_name: str):
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

    def get_drawing_objects_settings(self, plot_name: str):
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

    @staticmethod
    def allow_individual_limit_change(plot_drawing_objects: dict):
        """
        Checks if user requests setting of individual limits for y_axis
        :param plot_drawing_objects:
        :return:
        """

        min_max_indices = np.where(
            np.array([plot_drawing_objects['y_min'] is None, plot_drawing_objects['y_max'] is None]))[0]
        allow_changes = len(min_max_indices) == 1

        return allow_changes, min_max_indices

    @staticmethod
    def adjust_individual_limits(plot_drawing_objects, min_max_indices, vec):

        fcn_list = [np.min, np.max]
        compute_limit_tuple_indices = [0, 1]

        compute_limit_fcn = fcn_list[min_max_indices[0]]
        compute_limit_tuple_idx = compute_limit_tuple_indices[min_max_indices[0]]

        pdo = plot_drawing_objects
        plot_item = pdo['plot_item_ref']()
        limit = compute_limit_fcn(vec)
        y_range = [pdo['y_min'], pdo['y_max']]
        y_range[compute_limit_tuple_idx] = limit
        plot_item.setYRange(*y_range)

    def __show_all_plots_handler(self, _mutex=None):

        for plotName in list(self.plotData.keys()):
            if self.plotData[plotName][self.dirtyFlagIndex]:
                if plotName in list(self.plotDrawingObjects.keys()):
                    x_vec = self.plotData[plotName][0]
                    y_vec = self.plotData[plotName][1]

                    allow_changes, min_max_indices = self.allow_individual_limit_change(
                        plot_drawing_objects=self.plotDrawingObjects[plotName])

                    if allow_changes:
                        self.adjust_individual_limits(
                            plot_drawing_objects=self.plotDrawingObjects[plotName], min_max_indices=min_max_indices,
                            vec=y_vec)

                    plot_obj = self.plotDrawingObjects[plotName]['curve']

                    set_data_fcn = self.plotDrawingObjects[plotName]['SetData']
                    set_data_fcn(plot_obj, x_vec, y_vec)

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
