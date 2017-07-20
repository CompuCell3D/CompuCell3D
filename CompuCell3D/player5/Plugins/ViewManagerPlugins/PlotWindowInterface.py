# -*- coding: utf-8 -*-
# from PyQt5 import Qt
# from PyQt5.Qt import *

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import numpy as np

import warnings

try:
    import webcolors as wc
except ImportError:
    warnings.warn('Could not find webcolors. Run "pip install webcolors" to fix this', RuntimeWarning)

import pyqtgraph as pg
# pg.setConfigOption('background', 'w')
import pyqtgraph.exporters

import PlotManagerSetup
import os, Configuration
from enums import *

MODULENAME = '---- PlotManager.py: '

PLOT_TYPE_POSITION = 3
(XYPLOT, HISTOGRAM, BARPLOT) = range(0, 3)
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
    showHistPlotSignal = QtCore.pyqtSignal(str, QtCore.QMutex)
    showAllHistPlotsSignal = QtCore.pyqtSignal(QtCore.QMutex)

    showBarCurvePlotSignal = QtCore.pyqtSignal(str, QtCore.QMutex)
    showAllBarCurvePlotsSignal = QtCore.pyqtSignal(QtCore.QMutex)

    # savePlotAsPNG has to emit signal with locking mutex to work correctly
    savePlotAsPNGSignal = QtCore.pyqtSignal(str, int, int, QtCore.QMutex)

    setTitleSignal = QtCore.pyqtSignal(str)
    setTitleSizeSignal = QtCore.pyqtSignal(int)
    setPlotBackgroundColorSignal = QtCore.pyqtSignal(str)

    addPlotSignal = QtCore.pyqtSignal(object)

    def __init__(self, _plotWindow=None):
        # PlotManagerSetup.PlotWindowInterfaceBase.__init__(self,_plotWindow)
        QtCore.QObject.__init__(self, None)
        if _plotWindow:
            self.plotWindow = _plotWindow
            import weakref
            self.plotWindow.plotInterface = weakref.ref(self)
            self.pW = self.plotWindow.plotWidget

        self.plot_params = self.plotWindow.getPlotParams()

        self.plotData = {}
        self.plotHistData = {}
        self.plotDrawingObjects = {}
        self.initSignalsAndSlots()
        self.plotWindowInterfaceMutex = QtCore.QMutex()
        self.dirtyFlagIndex = 2  # this is the index of the flag tha is used to signal wheather the data has been modified or not
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

    def getQWTPLotWidget(self):  # returns native QWT widget to be manipulated by expert users
        return self.plotWindow

    def initSignalsAndSlots(self):
        self.showAllPlotsSignal.connect(self.__showAllPlots)
        self.showPlotSignal.connect(self.__showPlot)
        self.showAllHistPlotsSignal.connect(self.__showAllHistPlots)
        self.showHistPlotSignal.connect(self.__showHistPlot)
        self.showAllBarCurvePlotsSignal.connect(self.__showAllBarCurvePlots)
        self.showBarCurvePlotSignal.connect(self.__showBarCurvePlot)
        self.savePlotAsPNGSignal.connect(self.__savePlotAsPNG)

        self.setTitleSignal.connect(self.setTitleHandler)
        self.setTitleSizeSignal.connect(self.setTitleSizeHandler)
        self.setPlotBackgroundColorSignal.connect(self.setPlotBackgroundColorHandler)
        self.addPlotSignal.connect(self.addPlotHandler)

    def clear(self):
        # self.pW.clear()
        self.pW.detachItems()

    def replot(self):
        self.pW.replot()

    def setTitleHandler(self, _title):
        self.title = str(_title)
        self.pW.setTitle(_title)

    def setTitle(self, _title):
        self.title = str(_title)
        self.setTitleSignal.emit(_title)
        # self.pW.setTitle(_title)

    def setTitleSizeHandler(self, _size):
        print 'setTitleSizeHandler'
        # title = self.pW.titleLabel()
        # print 'title=',title
        # font = title.font()
        # font.setPointSize(_size)
        # title.setFont(font)
        # self.pW.setTitle(title)

    def setTitleSize(self, _size):
        self.setTitleSizeSignal.emit(_size)

    def setTitleColor(self, _colorName):
        try:
            title = self.pW.title()
            title.setColor(QColor(_colorName))
            self.pW.setTitle(title)
        except:
            raise RuntimeError('setTitleColor function is not supported in Player 5')

    def setPlotBackgroundColorHandler(self, _colorName):
        print '_colorName=', _colorName
        # self.pW.setCanvasBackground(QColor(_colorName))

    def setPlotBackgroundColor(self, _colorName):
        self.setPlotBackgroundColorSignal.emit(_colorName)
        self.pW.getViewBox().setBackgroundColor((255, 255, 255, 255))

        # self.pW.setCanvasBackground(QColor(_colorName))

    def addAutoLegend(self, _position="bottom"):
        pass

    def setDataDefault(self, plot_obj, x, y):
        plot_obj.setData(x, y)

    def setDataBarGraphItem(self, plot_obj, x, y):
        plot_obj.setOpts(x=x, height=y)

    def addPlot(self, _plotName, _style="Lines", _color='white', _size=3, _alpha=255):
        plot_param_dict = {
            '_plotName': _plotName,
            '_style': _style,
            '_color': _color,
            '_size': _size,
            '_alpha': _alpha
        }

        self.addPlotSignal.emit(plot_param_dict)

    # def addPlotHandler(self, _plotName, _style="Lines", _color='white', _size=3, _alpha=255):
    def addPlotHandler(self, plot_param_dict):

        _plotName = plot_param_dict['_plotName']
        _style = plot_param_dict['_style']
        _color = plot_param_dict['_color']
        _size = plot_param_dict['_size']
        _alpha = plot_param_dict['_alpha']

        add_legend = False
        try:
            add_legend = self.plot_params['legend']
        except KeyError:
            pass

        if add_legend and not self.legend_added:
            self.pW.addLegend()
            self.legend_added = True

        alpha = abs(int(_alpha)) % 256

        yd, xd = np.array([], dtype=np.float), np.array([], dtype=np.float)

        setData_fcn = self.setDataDefault

        color = wc.name_to_rgb(_color) + (alpha,)
        if _style.lower() == 'dots':
            plotObj = self.pW.plot(y=yd, x=xd, pen=(0, 0, 0), symbolBrush=color, symbolSize=_size, name=_plotName)


        elif _style.lower() == 'lines':
            pen = pg.mkPen(color=color, width=_size)
            plotObj = self.pW.plot(y=yd, x=xd, pen=pen, name=_plotName)


        elif _style.lower() == 'steps':
            xd, yd = np.array([0, .00001], dtype=np.float), np.array([1], dtype=np.float)
            # pen = pg.mkPen(color=color, width=_size)
            # plotObj = self.pW.plot(y=yd, x=xd, pen=pen,stepMode=True)

            plotObj = pg.PlotCurveItem(xd, yd, stepMode=True, fillLevel=0, brush=color, name=_plotName)

            self.pW.addItem(plotObj)
            # plt1.addItem(curve)

        elif _style.lower() == 'bars':

            plotObj = pg.BarGraphItem(x=xd, height=yd, width=_size, brush=color, name=_plotName)
            setData_fcn = self.setDataBarGraphItem
            self.pW.addItem(plotObj)

        else:  # dots is the default
            plotObj = self.pW.plot(y=yd, x=xd, pen=(0, 0, 0), symbolBrush=color, symbolSize=_size, name=_plotName)

        self.plotData[_plotName] = [xd, yd, False, XYPLOT, False]
        self.plotDrawingObjects[_plotName] = {'curve': plotObj,
                                              'LineWidth': _size,
                                              'LineColor': _color,
                                              'Style': _style,
                                              'SetData': setData_fcn}

    def addGrid(self):
        pass

    def eraseAllData(self):

        self.cleanAllContainers()

        for name, data in self.plotData.iteritems():
            data[self.dirtyFlagIndex] = True

        self.eraseAllFlag = True

    def cleanAllContainers(self):

        for name, data in self.plotData.iteritems():
            # todo implement special handling for "steps" or switch to bars instead of steps
            if data[0].shape[0] == data[1].shape[0] + 1:  # steps
                data[0], data[1] = np.array([0, .00001], dtype=np.float), np.array([0], dtype=np.float)
                data[4] = False
            else:
                data[0], data[1] = np.array([], dtype=np.float), np.array([], dtype=np.float)
            data[self.dirtyFlagIndex] = True

    def eraseData(self, _plotName):
        plotType = self.plotData[_plotName][PLOT_TYPE_POSITION]
        self.plotData[_plotName] = [array([], dtype=double), array([], dtype=double), False, plotType]

    def addDataSeries(self, _plotName, _x_vec, _y_vec):

        if not isinstance(_x_vec, (list, tuple, np.ndarray)):
            raise RuntimeError('addDataSeries: _x_vec has to be a list, tuple or 1D numpe array')

        if not isinstance(_y_vec, (list, tuple, np.ndarray)):
            raise RuntimeError('addDataSeries: _y_vec has to be a list, tuple or 1D numpe array')

        if len(_x_vec) != len(_y_vec):
            raise RuntimeError('addDataSeries: _x_vec and _y_vec have to be of the same length')

        for x, y in zip(_x_vec, _y_vec):
            self.addDataPoint(_plotName=_plotName, _x=x, _y=y)

    def addDataPoint(self, _plotName, _x, _y):

        if not _plotName in self.plotData.keys():
            return

        if self.eraseAllFlag:
            self.cleanAllContainers()
            self.eraseAllFlag = False

        currentLength = len(self.plotData[_plotName][0])

        x_vec = self.plotData[_plotName][0]
        y_vec = self.plotData[_plotName][1]

        plot_obj = self.plotDrawingObjects[_plotName]
        style = plot_obj['Style']
        if style.lower() == 'steps':
            # processing first first point of the histogram
            if not self.plotData[_plotName][4]:

                x_vec = np.array([_x, _x + 1], dtype=np.float)
                y_vec = np.array([_y], dtype=np.float)

                self.plotData[_plotName][0] = x_vec
                self.plotData[_plotName][1] = y_vec
                self.plotData[_plotName][4] = True
            else:
                self.plotData[_plotName][0] = np.append(self.plotData[_plotName][0], [_x])
                self.plotData[_plotName][1] = np.append(self.plotData[_plotName][1], [_y])
                self.plotData[_plotName][0][-2] = _x
                self.plotData[_plotName][0][-1] = _x + 1

        else:
            self.plotData[_plotName][0] = np.append(self.plotData[_plotName][0], [_x])
            self.plotData[_plotName][1] = np.append(self.plotData[_plotName][1], [_y])

        self.plotData[_plotName][self.dirtyFlagIndex] = True

    def getDrawingObjectsSettings(self, _plotName):
        if _plotName in self.plotDrawingObjects.keys():
            return self.plotDrawingObjects[_plotName]
        else:
            return None

    def changePlotProperty(self, _plotName, _property, _value):
        raise RuntimeError(
            '"changePlotProperty" is not supported in Player 5. It appears thst you are using old-style syntax that is no longer supported.')
        # self.plotDrawingObjects[_plotName][_property] = _value

    def setXAxisTitle(self, _title):
        pass

    def setYAxisTitle(self, _title):
        pass

    def setXAxisTitleSize(self, _size):
        pass

    def setXAxisTitleColor(self, _colorName):
        pass

    def setYAxisTitleSize(self, _size):

        pass

    def setYAxisTitleColor(self, _colorName):

        pass

    def setXAxisLogScale(self):
        pass

    def setYAxisLogScale(self):
        pass

    def setYAxisScale(self, _lower=0.0, _upper=100.0):
        pass

    def setXAxisScale(self, _lower=0.0, _upper=100.0):
        pass

    def showPlot(self, _plotName):
        pass

    def savePlotAsPNG(self, _fileName, _sizeX=400, _sizeY=400):
        self.plotWindowInterfaceMutex.lock()
        self.savePlotAsPNGSignal.emit(_fileName, _sizeX, _sizeY, self.plotWindowInterfaceMutex)

    def __savePlotAsPNG(self, _fileName, _sizeX, _sizeY, _mutex):

        warnings.warn('Player 5 does not allow scaling of plot screenshots. If this feature is required,'
                      'it is best to save plot data and render it separately in a full-featured plotting package such as Matpotlib or pyqtgraph. '
                      'CompuCell3D provides only basic plotting capabilities', RuntimeWarning)

        fileName = str(_fileName)
        #        pixmap=QPixmap(_sizeX,_sizeY)  # worked on Windows, but not Linux/OSX
        #        pixmap.fill(QColor("white"))

        # # WORKS OK todo
        # exporter = pg.exporters.ImageExporter(self.pW)
        # exporter.parameters()['width'] = 1000
        # exporter.parameters()['height'] = 1000
        #
        # exporter.export(_fileName)
        #

        # for plot_name, plot_obj_dict in self.plotDrawingObjects.iteritems():
        #     plot_obj = plot_obj_dict['curve']
        #     exporter = pg.exporters.ImageExporter(plot_obj)
        #     exporter.parameters()['width'] = 1000
        #     exporter.parameters()['height'] = 1000
        #     exporter.export(_fileName)
        # self.plotDrawingObjects[_plotName] = {'curve':plotObj,
        #                                       'LineWidth':_size,
        #                                       'LineColor': _color,
        #                                       'Style':_style,
        #                                       'SetData':setData_fcn}
        #
        #
        #
        #
        # import pyqtgraph.exporters
        # exporter = pg.exporters.ImageExporter(self.pW)
        # exporter.parameters()['width'] = 1000
        #
        # pixmap_array = self.pW.renderToArray((_sizeX,_sizeY))
        #
        pixmap = QPixmap(self.pW.size())
        painter = QPainter()
        painter.begin(pixmap)
        self.pW.render(painter, self.pW.sceneRect())
        pixmap.save(_fileName)
        painter.end()
        #
        # imgmap = QImage(_sizeX, _sizeY, QImage.Format_ARGB32)
        # # imgmap.fill(Qt.white)
        # imgmap.fill(
        #     qRgba(255, 255, 255, 255))  # solid white background (should probably depend on user-chosen colors though)
        #
        # self.pW.print_(imgmap)
        # # following seems pretty crude, but keep in mind user can change Prefs anytime during sim
        # # # #         if Configuration.getSetting("OutputToProjectOn"):
        # # # #             outDir = str(Configuration.getSetting("ProjectLocation"))
        # # # #         else:
        # # # #             outDir = str(Configuration.getSetting("OutputLocation"))
        #
        # import CompuCellSetup
        # outDir = CompuCellSetup.getSimulationOutputDir()
        #
        # outfile = os.path.join(outDir, fileName)
        # #        print '--------- savePlotAsPNG: outfile=',outfile
        # imgmap.save(outfile, "PNG")
        _mutex.unlock()

    def writeOutHeader(self, _file, _plotName, _outputFormat=LEGACY_FORMAT):

        if _outputFormat == LEGACY_FORMAT:

            _file.write(_plotName + '\n')
            return 0  # field width

        elif _outputFormat == CSV_FORMAT:

            plotName = _plotName.replace(' ', '_')

            fieldSize = len(plotName) + 2  # +2 is for _x or _y
            if MAX_FIELD_LEGTH > fieldSize:
                fieldSize = MAX_FIELD_LEGTH

            fmt = ''
            fmt += '{0:>' + str(fieldSize) + '},'
            fmt += '{1:>' + str(fieldSize) + '}\n'

            _file.write(fmt.format(plotName + '_x', plotName + '_y'))

            return fieldSize

        else:
            raise LookupError(
                MODULENAME + " writeOutHeader :" + "Requested output format: " + outputFormat + " does not exist")

    def savePlotAsData(self, _fileName, _outputFormat=LEGACY_FORMAT):
        # PLOT_TYPE_POSITION=3
        # (XYPLOT,HISTOGRAM,BARPLOT)=range(0,3)

        import CompuCellSetup
        outDir = CompuCellSetup.getSimulationOutputDir()

        outfile = os.path.join(outDir, _fileName)
        # print MODULENAME,'  savePlotAsData():   outfile=',outfile
        fpout = open(outfile, "w")
        # print MODULENAME,'  self.plotData= ',self.plotData

        for plotName, plotData in self.plotData.iteritems():
            # fpout.write(plotName+ '\n')
            fieldSize = self.writeOutHeader(_file=fpout, _plotName=plotName, _outputFormat=_outputFormat)

            xvals = plotData[0]
            yvals = plotData[1]
            # print MODULENAME,'  savePlotAsData():   xvals=',xvals
            # print MODULENAME,'  savePlotAsData():   yvals=',yvals
            if _outputFormat == LEGACY_FORMAT:
                if plotData[PLOT_TYPE_POSITION] == XYPLOT or plotData[PLOT_TYPE_POSITION] == BARPLOT:
                    for jdx in range(len(xvals)):

                        # we need to intercept Index Error because steps data appends extra data point in x array
                        try:
                            xyStr = "%f  %f\n" % (xvals[jdx], yvals[jdx])
                            fpout.write(xyStr)
                        except IndexError:
                            pass

                elif plotData[PLOT_TYPE_POSITION] == HISTOGRAM:
                    for jdx in range(len(xvals) - 1):

                        # we need to intercept Index Error because steps data appends extra data point in x array
                        try:
                            xyStr = "%f  %f\n" % (xvals[jdx], yvals[jdx])
                            fpout.write(xyStr)
                        except IndexError:
                            pass

            elif _outputFormat == CSV_FORMAT:
                fmt = ''
                fmt += '{0:>' + str(fieldSize) + '},'
                fmt += '{1:>' + str(fieldSize) + '}\n'

                if plotData[PLOT_TYPE_POSITION] == XYPLOT or plotData[PLOT_TYPE_POSITION] == BARPLOT:
                    for jdx in range(len(xvals)):

                        # we need to intercept Index Error because steps data appends extra data point in x array
                        try:
                            xyStr = fmt.format(xvals[jdx], yvals[jdx])
                            # "%f  %f\n" % (xvals[jdx],yvals[jdx])
                            fpout.write(xyStr)
                        except IndexError:
                            pass
                elif plotData[PLOT_TYPE_POSITION] == HISTOGRAM:
                    for jdx in range(len(xvals) - 1):

                        # we need to intercept Index Error because steps data appends extra data point in x array

                        try:
                            xyStr = fmt.format(xvals[jdx], yvals[jdx])
                            # xyStr = "%f  %f\n" % (xvals[jdx],yvals[jdx])
                            fpout.write(xyStr)
                        except IndexError:
                            pass



            else:
                raise LookupError(
                    MODULENAME + " savePlotAsData :" + "Requested output format: " + outputFormat + " does not exist")
            fpout.write('\n')  # separating data series by a line

        fpout.close()

    def __showPlot(self, plotName, _mutex=None):
        _plotName = str(plotName)
        if (not _plotName in self.plotData.keys()) or (not _plotName in self.plotDrawingObjects.keys()):
            return
        if not self.plotData[_plotName][self.dirtyFlagIndex]:
            return
        drawingObjects = self.plotDrawingObjects[_plotName]
        drawingObjects["curve"].attach(self.pW)
        drawingObjects["curve"].setPen(QPen(QColor(drawingObjects["LineColor"]), drawingObjects["LineWidth"]))
        drawingObjects["curve"].setData(self.plotData[_plotName][0], self.plotData[_plotName][1])
        self.plotData[_plotName][self.dirtyFlagIndex] = False
        if self.autoLegendFlag and not self.legendSetFlag:
            self.legend = Qwt.QwtLegend()
            self.legend.setFrameStyle(QFrame.Box | QFrame.Sunken)
            self.legend.setItemMode(Qwt.QwtLegend.ClickableItem)
            self.pW.insertLegend(self.legend, self.legendPosition)
            self.legendSetFlag = True
        self.pW.replot()
        self.plotWindowInterfaceMutex.unlock()

    def showAllPlots(self):
        self.plotWindowInterfaceMutex.lock()
        self.showAllPlotsSignal.emit(self.plotWindowInterfaceMutex)

    def __showAllPlots(self, _mutex=None):

        for plotName in self.plotData.keys():
            if self.plotData[plotName][self.dirtyFlagIndex]:
                if plotName in self.plotDrawingObjects.keys():
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

    def showAllHistPlots(self):
        self.plotWindowInterfaceMutex.lock()
        self.showAllHistPlotsSignal.emit(self.plotWindowInterfaceMutex)

    def __showHistPlot(self, plotName, _mutex=None):
        _plotName = str(plotName)
        print _plotName
        self.histogram.attach(self.pW)
        self.pW.replot()
        self.plotWindowInterfaceMutex.unlock()

    def __showAllHistPlots(self, _mutex=None):
        for hist in self.plotHistData.values():
            hist.attach(self.pW)
        self.pW.replot()
        _mutex.unlock()

    def addHistogram(self, plot_name, value_array, number_of_bins):

        (values, intervals) = np.histogram(value_array, bins=number_of_bins)

        self.plotData[plot_name][0] = intervals
        self.plotData[plot_name][1] = values

        self.plotData[plot_name][self.dirtyFlagIndex] = True

        # self.addHistPlotData(_plotName=plot_name, _values=values, _intervals=intervals)

    def addHistPlotData(self, _plotName, _values, _intervals):
        # print 'addHistPlotData'
        # print '_values=',_values
        # print '_intervals=',_intervals
        # self.plotData[_plotName]=[array([],dtype=double),array([],dtype=double),False]

        self.plotData[str(_plotName)] = [_intervals, _values, False, HISTOGRAM]

        intervals = []
        valLength = len(_values)
        values = Qwt.QwtArrayDouble(valLength)
        for i in range(valLength):
            # width = _intervals[i+1]-_intervals[i]+2
            intervals.append(Qwt.QwtDoubleInterval(_intervals[i], _intervals[
                i + 1]));  # numpy automcatically adds extra element for edge
            values[i] = _values[i]

        self.plotHistData[_plotName].setData(Qwt.QwtIntervalData(intervals, values))

    # todo OK
    # def addHistPlotData(self, _plotName, _values, _intervals):
    #     # print 'addHistPlotData'
    #     # print '_values=',_values
    #     # print '_intervals=',_intervals
    #     # self.plotData[_plotName]=[array([],dtype=double),array([],dtype=double),False]
    #
    #     self.plotData[str(_plotName)] = [_intervals, _values, False, HISTOGRAM]
    #
    #     intervals = []
    #     valLength = len(_values)
    #     values = Qwt.QwtArrayDouble(valLength)
    #     for i in range(valLength):
    #         # width = _intervals[i+1]-_intervals[i]+2
    #         intervals.append(Qwt.QwtDoubleInterval(_intervals[i], _intervals[
    #             i + 1]));  # numpy automcatically adds extra element for edge
    #         values[i] = _values[i]
    #
    #     self.plotHistData[_plotName].setData(Qwt.QwtIntervalData(intervals, values))

    def addHistPlot(self, _plotName, _r=100, _g=100, _b=0, _alpha=255):
        return
        self.plotHistData[_plotName] = HistogramItem()
        self.plotHistData[_plotName].setColor(QColor(_r, _g, _b, _alpha))

    def addHistogramPlot(self, _plotName, _color='blue', _alpha=255):
        self.addPlot(_plotName=_plotName, _style='Steps', _color=_color, _size=1.0, _alpha=_alpha)
        # self.plotHistData[_plotName] = HistogramItem()
        # color = QColor(_color)
        # color.setAlpha(_alpha)
        # self.plotHistData[_plotName].setColor(color)
        #
        # # def setHistogramColor(self,):
        # # self.histogram.setColor(QColor(_colorName))

    # def addHistogramPlot(self, _plotName, _color='blue', _alpha=255):
    #     self.plotHistData[_plotName] = HistogramItem()
    #     color = QColor(_color)
    #     color.setAlpha(_alpha)
    #     self.plotHistData[_plotName].setColor(color)
    #
    #     # def setHistogramColor(self,):
    #     # self.histogram.setColor(QColor(_colorName))

    def setHistogramColor(self, _colorName=None, _r=100, _g=100, _b=0, _alpha=255):
        if _colorName != None:
            # self.histogram.setColor(QColor(_colorName))
            self.plotHistData[_plotName].setColor(QColor(_colorName))
        else:
            # self.histogram.setColor(QColor(_r,_g,_b,_alpha))
            self.plotHistData[_plotName].setColor(QColor(_r, _g, _b, _alpha))

    def setHistogramView(self):
        self.histogram = HistogramItem()
        self.histogram.setColor(Qt.darkCyan)

        numValues = 20
        intervals = []
        values = Qwt.QwtArrayDouble(numValues)

        pos = 0.0
        for i in range(numValues):
            width = 5 + random.randint(0, 4)
            value = random.randint(0, 99)
            intervals.append(Qwt.QwtDoubleInterval(pos, pos + width));
            values[i] = value
            pos += width

            # self.histogram.setData(Qwt.QwtIntervalData(intervals, values))

    def showHistogram(self):
        self.plotWindowInterfaceMutex.lock()
        self.showAllHistPlotsSignal.emit(self.plotWindowInterfaceMutex)

    def addBarPlotData(self, _values, _positions, _width=1):

        raise RuntimeError('addBarPlotData is not supported in Player 5. Instead Please use regular plots and '
                           'specify plotting style as "bars" - '
                           'self.pW.addPlot(_plotName="GDP",_color="red",_style="bars", _size=0.5). it is a good idea to '
                           'clean plot before plotting new series by calling e.g. self.pW.eraseAllData()')

        self.plotData[self.title] = [_positions, _values, False, BARPLOT]

        for bar in self.pW.itemList():
            if isinstance(bar, BarCurve):
                bar.detach()

        for i in range(len(_values)):
            self.barplot = BarCurve()
            self.barplot.attach(self.pW)
            self.barplot.setData([float(_positions[i]), float(_positions[i] + _width)], [0, float(_values[i])])

    def setBarPlotView(self):
        # do nothing
        pass

    def showAllBarCurvePlots(self):
        self.plotWindowInterfaceMutex.lock()
        self.showAllBarCurvePlotsSignal.emit(self.plotWindowInterfaceMutex)

    def __showBarCurvePlot(self, _plotName, _mutex=None):
        plotName = str(_plotName)
        self.pW.replot()
        self.plotWindowInterfaceMutex.unlock()

    def __showAllBarCurvePlots(self, _mutex=None):
        if self.barplot is not None:
            self.barplot.attach(self.pW)
        self.pW.replot()
        _mutex.unlock()
