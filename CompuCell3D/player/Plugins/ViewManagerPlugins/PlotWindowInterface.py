# -*- coding: utf-8 -*-
from PyQt5 import Qt
from PyQt5.Qt import *

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
import numpy as np

# import PyQt4.Qwt5 as Qwt
# from PyQt4.Qwt5.anynumpy import *

import PlotManagerSetup
import os, Configuration
from enums import *

MODULENAME = '---- PlotManager.py: '

PLOT_TYPE_POSITION = 3
(XYPLOT, HISTOGRAM, BARPLOT) = range(0, 3)
MAX_FIELD_LEGTH = 25


# Notice histogram and Bar Plot implementations need more work. They are functional but have a bit strange syntax and for Bar Plot we can only plot one series per plot

# class PlotWindowInterface(PlotManagerSetup.PlotWindowInterfaceBase,QtCore.QObject):
class PlotWindowInterface(QtCore.QObject):
    # showPlotSignal = QtCore.pyqtSignal( (QtCore.QString,QtCore.QMutex, ))
    # showBarCurvePlotSignal = QtCore.pyqtSignal( ('char*',QtCore.QMutex, ))
    # savePlotAsPNGSignal=QtCore.pyqtSignal( ('char*','int','int',QtCore.QMutex, )) #savePlotAsPNG has to emit signal with locking mutex to work correctly
    # showPlotSignal = QtCore.pyqtSignal( ('char*',QtCore.QMutex, ))
    # showHistPlotSignal = QtCore.pyqtSignal( ('char*',QtCore.QMutex, ))
    # showPlotSignal = QtCore.pyqtSignal( (QtCore.QString,QtCore.QMutex, ))
    # showAllPlotsSignal=QtCore.pyqtSignal( (QtCore.QMutex, ))
    # showHistPlotSignal = QtCore.pyqtSignal( (QtCore.QString,QtCore.QMutex, ))
    # showAllHistPlotsSignal=QtCore.pyqtSignal( (QtCore.QMutex, ))
    #
    # showBarCurvePlotSignal = QtCore.pyqtSignal( (QtCore.QString,QtCore.QMutex, ))
    # showAllBarCurvePlotsSignal=QtCore.pyqtSignal( (QtCore.QMutex, ))
    #
    # savePlotAsPNGSignal=QtCore.pyqtSignal( (QtCore.QString,'int','int',QtCore.QMutex, )) #savePlotAsPNG has to emit signal with locking mutex to work correctly


    # IMPORTANT: turns out that using QString in signals is essential to get correct behavior. for some reason using char * in signal may malfunction e.g. wrong string is sent to slot.

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

    def __init__(self, _plotWindow=None):
        # PlotManagerSetup.PlotWindowInterfaceBase.__init__(self,_plotWindow)
        QtCore.QObject.__init__(self, None)
        if _plotWindow:
            self.plotWindow = _plotWindow
            import weakref
            self.plotWindow.plotInterface = weakref.ref(self)
            self.pW = self.plotWindow.plotWidget

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

        self.barplot = None

        self.eraseAllFlag = False
        self.logScaleFlag = False
        self.title = ''

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
        title = self.pW.title()
        title.setColor(QColor(_colorName))
        self.pW.setTitle(title)

    def setPlotBackgroundColorHandler(self, _colorName):
        print '_colorName=',_colorName
        # self.pW.setCanvasBackground(QColor(_colorName))


    def setPlotBackgroundColor(self, _colorName):
        self.setPlotBackgroundColorSignal.emit(_colorName)
        self.pW.getViewBox().setBackgroundColor((255, 255, 255, 255))

        # self.pW.setCanvasBackground(QColor(_colorName))

    def addAutoLegend(self, _position="bottom"):
        self.autoLegendFlag = True

        # print "_position=",_position
        # sys.exit()

        if _position.lower() == "top":
            self.legendPosition = Qwt.QwtPlot.TopLegend
            # print "_position=",_position
            # sys.exit()

        elif _position.lower() == "bottom":
            self.legendPosition = Qwt.QwtPlot.BottomLegend
        elif _position.lower() == "left":
            self.legendPosition = Qwt.QwtPlot.LeftLegend
        elif _position.lower() == "right":
            self.legendPosition = Qwt.QwtPlot.RightLegend


            # self.legend = Qwt.QwtLegend()
            # self.legend.setFrameStyle(QFrame.Box | QFrame.Sunken)
            # self.legend.setItemMode(Qwt.QwtLegend.ClickableItem)

            # self.pW.insertLegend(legend, Qwt.QwtPlot.TopLegend)

    def addPlot(self, _plotName, _style="Lines", _color='black', _size=1):

        a = np.random.rand(10)
        # yd, xd = np.random.rand(10),np.random.rand(10)

        # yd, xd = np.random.rand(10), np.random.rand(10)
        yd, xd = np.array([], dtype=np.float), np.array([], dtype=np.float)
        self.plotData[_plotName] = [xd,yd,False, XYPLOT]
        plotObj = self.pW.plot(y=yd, x=xd)
        self.plotDrawingObjects[_plotName] = plotObj


        # p1.setData()

        # self.pW.addPlot('plot_name')

        # # self.plotData[_plotName]=[arange(0),arange(0),False]
        #
        # self.plotData[_plotName] = [array([], dtype=double), array([], dtype=double), False, XYPLOT]
        #
        # self.plotDrawingObjects[_plotName] = {"curve": Qwt.QwtPlotCurve(_plotName), "LineWidth": _size,
        #                                       "LineColor": _color}
        # plotStyle = getattr(Qwt.QwtPlotCurve, _style)
        # # self.plotDrawingObjects[_plotName]["curve"].setStyle(Qwt.QwtPlotCurve.Dots)
        # self.plotDrawingObjects[_plotName]["curve"].setStyle(plotStyle)

    def addGrid(self):
        grid = Qwt.QwtPlotGrid()
        grid.attach(self.pW)
        grid.setPen(QPen(Qt.black, 0, Qt.DotLine))

    def eraseAllData(self):

        self.cleanAllContainers()

        for name, data in self.plotData.iteritems():
            data[self.dirtyFlagIndex] = True

        self.eraseAllFlag = True

    def cleanAllContainers(self):

        for name, data in self.plotData.iteritems():
            data[0].resize(0)
            data[1].resize(0)
            data[self.dirtyFlagIndex] = True

    def eraseData(self, _plotName):
        plotType = self.plotData[_plotName][PLOT_TYPE_POSITION]
        self.plotData[_plotName] = [array([], dtype=double), array([], dtype=double), False, plotType]

    def addDataPoint(self, _plotName, _x, _y):

        if not _plotName in self.plotData.keys():
            return

        if self.eraseAllFlag:
            self.cleanAllContainers()
            self.eraseAllFlag = False

        currentLength = len(self.plotData[_plotName][0])

        self.plotData[_plotName][0] = np.append(self.plotData[_plotName][0],[_x])
        self.plotData[_plotName][1] = np.append(self.plotData[_plotName][1], [_y])

        # self.plotData[_plotName][0].resize(currentLength + 1)
        # self.plotData[_plotName][1].resize(currentLength + 1)
        #
        # self.plotData[_plotName][0][currentLength] = _x
        # self.plotData[_plotName][1][currentLength] = _y
        self.plotData[_plotName][self.dirtyFlagIndex] = True
        # print "self.plotData[_plotName][0]=",self.plotData[_plotName][0]
        # print "self.plotData[_plotName][1]=",self.plotData[_plotName][1]


        # if not _plotName in self.plotData.keys():
        #     return
        #
        # if self.eraseAllFlag:
        #     self.cleanAllContainers()
        #     self.eraseAllFlag = False
        #
        # currentLength = len(self.plotData[_plotName][0])
        # self.plotData[_plotName][0].resize(currentLength + 1)
        # self.plotData[_plotName][1].resize(currentLength + 1)
        #
        # self.plotData[_plotName][0][currentLength] = _x
        # self.plotData[_plotName][1][currentLength] = _y
        # self.plotData[_plotName][self.dirtyFlagIndex] = True
        # # print "self.plotData[_plotName][0]=",self.plotData[_plotName][0]
        # # print "self.plotData[_plotName][1]=",self.plotData[_plotName][1]

    def getDrawingObjectsSettings(self, _plotName):
        if _plotName in self.plotDrawingObjects.keys():
            return self.plotDrawingObjects[_plotName]
        else:
            return None

    def changePlotProperty(self, _plotName, _property, _value):
        self.plotDrawingObjects[_plotName][_property] = _value

    def setXAxisTitle(self, _title):
        self.pW.setAxisTitle(Qwt.QwtPlot.xBottom, _title)

    def setYAxisTitle(self, _title):
        self.pW.setAxisTitle(Qwt.QwtPlot.yLeft, _title)

    def setXAxisTitleSize(self, _size):

        title = self.pW.axisTitle(Qwt.QwtPlot.xBottom)
        font = title.font()
        font.setPointSize(_size)
        title.setFont(font)
        self.pW.setAxisTitle(Qwt.QwtPlot.xBottom, title)

    def setXAxisTitleColor(self, _colorName):

        title = self.pW.axisTitle(Qwt.QwtPlot.xBottom)
        title.setColor(QColor(_colorName))
        self.pW.setAxisTitle(Qwt.QwtPlot.xBottom, title)

    def setYAxisTitleSize(self, _size):

        title = self.pW.axisTitle(Qwt.QwtPlot.yLeft)
        font = title.font()
        font.setPointSize(_size)
        title.setFont(font)
        self.pW.setAxisTitle(Qwt.QwtPlot.yLeft, title)

    def setYAxisTitleColor(self, _colorName):

        title = self.pW.axisTitle(Qwt.QwtPlot.yLeft)
        title.setColor(QColor(_colorName))
        self.pW.setAxisTitle(Qwt.QwtPlot.yLeft, title)

    def setXAxisLogScale(self):
        self.pW.setAxisScaleEngine(Qwt.QwtPlot.xBottom, Qwt.QwtLog10ScaleEngine())
        self.logScaleFlag = True

    def setYAxisLogScale(self):
        self.pW.setAxisScaleEngine(Qwt.QwtPlot.yLeft, Qwt.QwtLog10ScaleEngine())
        self.logScaleFlag = True

    def setYAxisScale(self, _lower=0.0, _upper=100.0):
        self.pW.setAxisScale(Qwt.QwtPlot.yLeft, _lower, _upper)

    def setXAxisScale(self, _lower=0.0, _upper=100.0):
        self.pW.setAxisScale(Qwt.QwtPlot.xBottom, _lower, _upper)

    def showPlot(self, _plotName):
        self.plotWindowInterfaceMutex.lock()
        self.showPlotSignal.emit(QString(_plotName), self.plotWindowInterfaceMutex)

    def savePlotAsPNG(self, _fileName, _sizeX=400, _sizeY=400):
        self.plotWindowInterfaceMutex.lock()
        self.savePlotAsPNGSignal.emit(_fileName, _sizeX, _sizeY, self.plotWindowInterfaceMutex)

    def __savePlotAsPNG(self, _fileName, _sizeX, _sizeY, _mutex):
        fileName = str(_fileName)
        #        pixmap=QPixmap(_sizeX,_sizeY)  # worked on Windows, but not Linux/OSX
        #        pixmap.fill(QColor("white"))

        imgmap = QImage(_sizeX, _sizeY, QImage.Format_ARGB32)
        # imgmap.fill(Qt.white)
        imgmap.fill(
            qRgba(255, 255, 255, 255))  # solid white background (should probably depend on user-chosen colors though)

        self.pW.print_(imgmap)
        # following seems pretty crude, but keep in mind user can change Prefs anytime during sim
        # # #         if Configuration.getSetting("OutputToProjectOn"):
        # # #             outDir = str(Configuration.getSetting("ProjectLocation"))
        # # #         else:
        # # #             outDir = str(Configuration.getSetting("OutputLocation"))

        import CompuCellSetup
        outDir = CompuCellSetup.getSimulationOutputDir()

        outfile = os.path.join(outDir, fileName)
        #        print '--------- savePlotAsPNG: outfile=',outfile
        imgmap.save(outfile, "PNG")
        _mutex.unlock()

        # original implementation - does not really work unless we use signal slot mechanism
        # def savePlotAsPNG(self,_fileName,_sizeX=400,_sizeY=400):

    # #        pixmap=QPixmap(_sizeX,_sizeY)  # worked on Windows, but not Linux/OSX
    # #        pixmap.fill(QColor("white"))

    # imgmap = QImage(_sizeX, _sizeY, QImage.Format_ARGB32)
    # #imgmap.fill(Qt.white)
    # imgmap.fill(qRgba(255, 255, 255, 255)) # solid white background (should probably depend on user-chosen colors though)

    # self.pW.print_(imgmap)
    # # following seems pretty crude, but keep in mind user can change Prefs anytime during sim
    # if Configuration.getSetting("OutputToProjectOn"):
    # outDir = str(Configuration.getSetting("ProjectLocation"))
    # else:
    # outDir = str(Configuration.getSetting("OutputLocation"))
    # outfile = os.path.join(outDir,_fileName)
    # #        print '--------- savePlotAsPNG: outfile=',outfile
    # imgmap.save(outfile,"PNG")

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
                        xyStr = "%f  %f\n" % (xvals[jdx], yvals[jdx])
                        fpout.write(xyStr)
                elif plotData[PLOT_TYPE_POSITION] == HISTOGRAM:
                    for jdx in range(len(xvals) - 1):
                        xyStr = "%f  %f\n" % (xvals[jdx], yvals[jdx])
                        fpout.write(xyStr)

            elif _outputFormat == CSV_FORMAT:
                fmt = ''
                fmt += '{0:>' + str(fieldSize) + '},'
                fmt += '{1:>' + str(fieldSize) + '}\n'

                if plotData[PLOT_TYPE_POSITION] == XYPLOT or plotData[PLOT_TYPE_POSITION] == BARPLOT:
                    for jdx in range(len(xvals)):
                        xyStr = fmt.format(xvals[jdx], yvals[jdx])
                        # "%f  %f\n" % (xvals[jdx],yvals[jdx])
                        fpout.write(xyStr)
                elif plotData[PLOT_TYPE_POSITION] == HISTOGRAM:
                    for jdx in range(len(xvals) - 1):
                        xyStr = fmt.format(xvals[jdx], yvals[jdx])
                        # xyStr = "%f  %f\n" % (xvals[jdx],yvals[jdx])
                        fpout.write(xyStr)


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

                    drawingObjects = self.plotDrawingObjects[plotName]
                    drawingObjects.setData(x_vec,y_vec)
                    # drawingObjects["curve"].attach(self.pW)
                    # drawingObjects["curve"].setPen(
                    #     QPen(QColor(drawingObjects["LineColor"]), drawingObjects["LineWidth"]))
                    # drawingObjects["curve"].setData(self.plotData[plotName][0], self.plotData[plotName][1])
                    #
                    # #                    print "self.legendPosition=",self.legendPosition
                    # if self.autoLegendFlag and not self.legendSetFlag:
                    #     self.legend = Qwt.QwtLegend()
                    #     self.legend.setFrameStyle(QFrame.Box | QFrame.Sunken)
                    #     self.legend.setItemMode(Qwt.QwtLegend.ClickableItem)
                    #     self.pW.insertLegend(self.legend, self.legendPosition)
                    #     self.legendSetFlag = True

                    # self.pW.replot()
                    self.plotData[plotName][self.dirtyFlagIndex] = False


        # for plotName in self.plotData.keys():
        #     if self.plotData[plotName][self.dirtyFlagIndex]:
        #         if plotName in self.plotDrawingObjects.keys():
        #             drawingObjects = self.plotDrawingObjects[plotName]
        #             drawingObjects["curve"].attach(self.pW)
        #             drawingObjects["curve"].setPen(
        #                 QPen(QColor(drawingObjects["LineColor"]), drawingObjects["LineWidth"]))
        #             drawingObjects["curve"].setData(self.plotData[plotName][0], self.plotData[plotName][1])
        #
        #             #                    print "self.legendPosition=",self.legendPosition
        #             if self.autoLegendFlag and not self.legendSetFlag:
        #                 self.legend = Qwt.QwtLegend()
        #                 self.legend.setFrameStyle(QFrame.Box | QFrame.Sunken)
        #                 self.legend.setItemMode(Qwt.QwtLegend.ClickableItem)
        #                 self.pW.insertLegend(self.legend, self.legendPosition)
        #                 self.legendSetFlag = True
        #
        #             self.pW.replot()
        #             self.plotData[plotName][self.dirtyFlagIndex] = False
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
        import numpy
        (values, intervals) = numpy.histogram(value_array, bins=number_of_bins)
        self.addHistPlotData(_plotName=plot_name, _values=values, _intervals=intervals)

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

    def addHistPlot(self, _plotName, _r=100, _g=100, _b=0, _alpha=255):
        self.plotHistData[_plotName] = HistogramItem()
        self.plotHistData[_plotName].setColor(QColor(_r, _g, _b, _alpha))

    def addHistogramPlot(self, _plotName, _color='black', _alpha=255):
        self.plotHistData[_plotName] = HistogramItem()
        color = QColor(_color)
        color.setAlpha(_alpha)
        self.plotHistData[_plotName].setColor(color)

        # def setHistogramColor(self,):
        # self.histogram.setColor(QColor(_colorName))

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

