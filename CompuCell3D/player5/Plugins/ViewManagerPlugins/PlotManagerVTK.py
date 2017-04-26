# -*- coding: utf-8 -*-
#from chaco.api import ArrayPlotData,Plot,create_line_plot,add_default_axes,add_default_grids,OverlayPlotContainer
#from enable.api import Component, Container, Window
#from enable.api import Window as chacoWindow
#from pyface.qt import QtCore, QtGui

from PyQt4 import QtCore

#import PlotManagerSetup
import os, Configuration 

#from numpy import array,double
#from numpy import linspace,pi,sin

import vtk
#import math
from  Graphics.PlotFrameWidgetVTK import PlotFrameWidget
from  Graphics.GraphicsFrameWidget import GraphicsFrameWidget


MODULENAME='----- PlotManagerVTK.py: '


#=====================================================================
# class PlotWindowInterface(PlotManagerSetup.PlotWindowInterfaceBase, QtCore.QObject):
class PlotWindowInterface(QtCore.QObject):
    showPlotSignal = QtCore.pyqtSignal( (QtCore.QString,QtCore.QMutex, ))
    # showPlotSignal = QtCore.pyqtSignal( ('char*',QtCore.QMutex, ))
    showAllPlotsSignal = QtCore.pyqtSignal( (QtCore.QMutex, ))
#    showHistPlotSignal = QtCore.pyqtSignal( ('char*',QtCore.QMutex, ))
#    showAllHistPlotsSignal = QtCore.pyqtSignal( (QtCore.QMutex, ))
#    showBarCurvePlotSignal = QtCore.pyqtSignal( ('char*',QtCore.QMutex, ))
#    showAllBarCurvePlotsSignal = QtCore.pyqtSignal( (QtCore.QMutex, ))
    
    def __init__(self,_plotWindow=None):
        # PlotManagerSetup.PlotWindowInterfaceBase.__init__(self,_plotWindow)
        QtCore.QObject.__init__(self, None)
        print MODULENAME,' __init__():  _plotWindow=',_plotWindow
        if _plotWindow:
            self.plotWindow = _plotWindow
#            self.pW = self.plotWindow.plotWidget        
            self.pW = self.plotWindow.qvtkWidget        
            print MODULENAME,'PlotWindowInterface: __init__():  self.pW=',self.pW

        self.chart = vtk.vtkChartXY()
        self.plotData={}
        self.plotHistData={}
        self.plotDrawingObjects={}
        self.initSignalsAndSlots()
        self.plotWindowInterfaceMutex = QtCore.QMutex()
        self.dirtyFlagIndex=2 # this is the index of the flag that is used to signal whether the data has been modified or not
        self.autoLegendFlag=False
        self.legendSetFlag=False
        
        self.eraseAllFlag=False
        self.logScaleFlag=False
        
#    def getQWTPLotWidget(self): # returns native QWT widget to be manipulated by expert users
#        return self.plotWindow        
        
    def initSignalsAndSlots(self):
        self.showAllPlotsSignal.connect(self.__showAllPlots)
        self.showPlotSignal.connect(self.__showPlot)
#        self.showAllHistPlotsSignal.connect(self.__showAllHistPlots)
#        self.showHistPlotSignal.connect(self.__showHistPlot)
#        self.showAllBarCurvePlotsSignal.connect(self.__showAllBarCurvePlots)
#        self.showBarCurvePlotSignal.connect(self.__showBarCurvePlot)
    
    def clear(self):
        # self.pW.clear()
        self.pW.detachItems()
        
    def replot(self):
        self.pW.replot()
        
    def setTitle(self,_title):
        print MODULENAME,'PlotWindowInterface: setTitle():  self.chart=',self.chart,', _title=',_title
        self.chart.SetTitle(_title)   # 'PlotWindowInterface' object has no attribute 'chart'
#        self.pW.setTitle(_title)       # QVTKRenderWindowInteractor has no attribute named setTitle
        
    def setTitleSize(self,_size):
        title = self.pW.title()
        font = title.font()
        font.setPointSize(_size)
        title.setFont(font)        
        self.pW.setTitle(title)
        
    def setTitleColor(self,_colorName):
        title=self.pW.title()
        title.setColor(QColor(_colorName))
        self.pW.setTitle(title)
        
    def setPlotBackgroundColor(self,_colorName):
        self.pW.setCanvasBackground(QColor(_colorName))

    def addAutoLegend(self,_position="bottom"): 
        self.autoLegendFlag=True
        
    def addPlot(self,_plotName,_style="Lines"):   # called directly from Steppable; add a (possibly more than one) plot to a plot window

        self.plotWindowInterfaceMutex.lock()
#        self.plotWindowMutex.lock()

#        return
#        print MODULENAME,'   addPlot():  _plotName= ',_plotName
#        import pdb; pdb.set_trace()
        
#        self.plotData[_plotName] = [array([],dtype=double),array([],dtype=double),False]  # 'array': from PyQt4.Qwt5.anynumpy import *
        
        self.chart = vtk.vtkChartXY()
#        self.chart.GetAxis(vtk.vtkAxis.LEFT).SetLogScale(True)
#        self.chart.GetAxis(vtk.vtkAxis.BOTTOM).SetLogScale(True)
#        self.numCharts += 1
        self.plotData[_plotName] = [self.chart]

        self.view = vtk.vtkContextView()
        self.ren = self.view.GetRenderer()
#        self.renWin = self.qvtkWidget.GetRenderWindow()
        self.renWin = self.pW.GetRenderWindow()
        self.renWin.AddRenderer(self.ren)

        # Create a table with some points in it
        self.table = vtk.vtkTable()

        self.arrX = vtk.vtkFloatArray()
        self.arrX.SetName("xarray")

        self.arrC = vtk.vtkFloatArray()
        self.arrC.SetName("yarray")

        numPoints = 5
        numPoints = 15
        inc = 7.5 / (numPoints - 1)

#        for i in range(0,numPoints):
#            self.arrX.InsertNextValue(i*inc)
#            self.arrC.InsertNextValue(math.cos(i * inc) + 0.0)

#        self.arrX.InsertNextValue(0.0)
#        self.arrC.InsertNextValue(0.0)
#        self.arrX.InsertNextValue(0.1)
#        self.arrC.InsertNextValue(0.1)

        self.table.AddColumn(self.arrX)
        self.table.AddColumn(self.arrC)

        # Now add the line plots with appropriate colors
        self.line = self.chart.AddPlot(0)
        self.line.SetInput(self.table,0,1)
        self.line.SetColor(0,0,255,255)
        self.line.SetWidth(1.0)


        self.view.GetRenderer().SetBackground([0.6,0.6,0.1])
        self.view.GetRenderer().SetBackground([1.0,1.0,1.0])
        self.view.GetScene().AddItem(self.chart)

        self.plotWindowInterfaceMutex.unlock()
#        self.plotWindowMutex.unlock()


    def eraseAllData(self):    
        self.cleanAllContainers()
        for name, data in self.plotData.iteritems():
            data[self.dirtyFlagIndex] = True
            
        self.eraseAllFlag = True
            
    def cleanAllContainers(self):
        for name,data in self.plotData.iteritems():
            data[0].resize(0)
            data[1].resize(0)          
            data[self.dirtyFlagIndex] = True

    def eraseData(self,_plotName):
        self.plotData[_plotName] = [array([],dtype=double),array([],dtype=double),False]

    def addDataPoint(self,_plotName, _x,_y):        #  <rwh---------------------
        print MODULENAME,"addDataPoint():  _plotName=",_plotName,", _x,_y=",_x,_y
#        print self.plotData[_plotName] = [self.chart]
        print MODULENAME,"addDataPoint():  self.plotData=",self.plotData
        vtkChart = self.plotData[_plotName][0]
#        print MODULENAME,"addDataPoint():  vtkChart=",vtkChart
        xarray = vtkChart.GetPlot(0).GetInput().GetColumn(0)
        print MODULENAME,"addDataPoint():  type(xarray)=",type(xarray)
        yarray = vtkChart.GetPlot(0).GetInput().GetColumnByName("yarray")
#        print MODULENAME,"addDataPoint():  yarray=",yarray

        xarray.InsertNextValue(_x)
        yarray.InsertNextValue(_y)

        print MODULENAME,"addDataPoint():  xarray.GetNumberOfTuples()=",xarray.GetNumberOfTuples()
        if xarray.GetNumberOfTuples() < 2:
            return

        plot = vtkChart.GetPlot(0)
        plot.Modified()
        vtkChart.RecalculateBounds()   # Necessary?  Apparently so.
        self.pW.Render()
#        self.chart.SetTitle("My awesome plot")   # 'PlotWindowInterface' object has no attribute 'chart'

        if not _plotName in self.plotData.keys():
            print MODULENAME,"addDataPoint():  1st return"
            return
            
        if self.eraseAllFlag:
            self.cleanAllContainers()            
            self.eraseAllFlag = False
            
#        currentLength = len(self.plotData[_plotName][0])    # --> TypeError: object of type 'vtkobject' has no len()
#        self.plotData[_plotName][0].resize(currentLength+1)
#        self.plotData[_plotName][1].resize(currentLength+1)
        
#        self.plotData[_plotName][0][currentLength] = _x
#        self.plotData[_plotName][1][currentLength] = _y
#        self.plotData[_plotName][self.dirtyFlagIndex]=True
        # print "self.plotData[_plotName][0]=",self.plotData[_plotName][0]
        # print "self.plotData[_plotName][1]=",self.plotData[_plotName][1]
        
       
    def getDrawingObjectsSettings(self,_plotName):
        if _plotName in self.plotDrawingObjects.keys():
            return self.plotDrawingObjects[_plotName]
        else:
            return None

    def changePlotProperty(self,_plotName,_property,_value):
        self.plotDrawingObjects[_plotName][_property]=_value
        
#    def setXAxisTitle(self,_title):


    def showPlot(self,_plotName):
        print MODULENAME,'  showPlot():   _plotName=',_plotName
        self.plotWindowInterfaceMutex.lock()
        self.showPlotSignal.emit(QString(_plotName),self.plotWindowInterfaceMutex)
        
    def __showPlot(self,plotName,_mutex=None):
        _plotName=QString(plotName)
        print MODULENAME,'  __showPlot():   self.plotData.keys()=',self.plotData.keys()
        print MODULENAME,'  __showPlot():   self.plotDrawingObjects.keys()=',self.plotDrawingObjects.keys()
#        if (not _plotName in self.plotData.keys() ) or (not _plotName in self.plotDrawingObjects.keys() ):                        
        if (not _plotName in self.plotData.keys() ) :                        
            print MODULENAME,'  __showPlot():   1st return'
            self.plotWindowInterfaceMutex.unlock()
            return
#        if not self.plotData[_plotName][self.dirtyFlagIndex]:                        
#            print MODULENAME,'  __showPlot():   not dirty, return'
#            self.plotWindowInterfaceMutex.unlock()
#            return

#        drawingObjects = self.plotDrawingObjects[_plotName]
#        drawingObjects["curve"].attach(self.pW)
#        drawingObjects["curve"].setPen(QPen(QColor(drawingObjects["LineColor"]), drawingObjects["LineWidth"]))
#        drawingObjects["curve"].setData(self.plotData[_plotName][0], self.plotData[_plotName][1])
#        self.plotData[_plotName][self.dirtyFlagIndex]=False

#        if self.autoLegendFlag and not self.legendSetFlag:
#            self.legend = Qwt.QwtLegend()
#            self.legend.setFrameStyle(QFrame.Box | QFrame.Sunken)
#            self.legend.setItemMode(Qwt.QwtLegend.ClickableItem)
#            self.pW.insertLegend(self.legend, self.legendPosition)                    
#            self.legendSetFlag=True
#        self.pW.replot()

        self.plotWindowInterfaceMutex.unlock()


    def showAllPlots(self):    
        self.plotWindowInterfaceMutex.lock()
        self.showAllPlotsSignal.emit(self.plotWindowInterfaceMutex)

    def __showAllPlots(self,_mutex=None):
        print MODULENAME,'  __showAllPlots():   self.plotDrawingObjects.keys()=',self.plotDrawingObjects.keys()
        for plotName in self.plotData.keys():
            print MODULENAME,'  __showAllPlots():  plotName=',plotName
            if self.plotData[plotName][self.dirtyFlagIndex]:
                if plotName in self.plotDrawingObjects.keys():                
                    print MODULENAME,'  __showAllPlots():  doing ',plotName
                    drawingObjects = self.plotDrawingObjects[plotName]    
                    drawingObjects["curve"].attach(self.pW)

#                    drawingObjects["curve"].attach(self.pW)
#                    drawingObjects["curve"].setPen(QPen(QColor(drawingObjects["LineColor"]), drawingObjects["LineWidth"]))
#                    drawingObjects["curve"].setData(self.plotData[plotName][0], self.plotData[plotName][1])                    
                    
#                    print "self.legendPosition=",self.legendPosition
#                    if self.autoLegendFlag and not self.legendSetFlag:
#                        self.legend = Qwt.QwtLegend()
#                        self.legend.setFrameStyle(QFrame.Box | QFrame.Sunken)
#                        self.legend.setItemMode(Qwt.QwtLegend.ClickableItem)
#                        self.pW.insertLegend(self.legend, self.legendPosition)                    
#                        self.legendSetFlag=True
                        
#                    self.pW.replot()
                    self.plotData[plotName][self.dirtyFlagIndex]=False
        _mutex.unlock()

#    def showAllHistPlots(self):    

    def setData(self, data):
        self.__data = data
        self.itemChanged()

    def data(self):
        return self.__data

    def setColor(self, color):
        if self.__color != color:
            self.__color = color
            self.itemChanged()

    def color(self):
        return self.__color

    def boundingRect(self):
        result = self.__data.boundingRect()
        if not result.isValid():
            return result
        if self.testHistogramAttribute(HistogramItem.Xfy):
            pass
        else:
            if result.bottom() < self.baseline():
                result.setBottom(self.baseline())
            elif result.top() > self.baseline():
                result.setTop(self.baseline())
        return result

    def rwh_draw(self, painter, xMap, yMap, rect):
        iData = self.data()
        painter.setPen(self.color())
        x0 = xMap.transform(self.baseline())
        y0 = yMap.transform(self.baseline())
        for i in range(iData.size()):
            if self.testHistogramAttribute(HistogramItem.Xfy):
                x2 = xMap.transform(iData.value(i))
                if x2 == x0:
                    continue

                y1 = yMap.transform(iData.interval(i).minValue())
                y2 = yMap.transform(iData.interval(i).maxValue())

                if y1 > y2:
                    y1, y2 = y2, y1
                    
                if  i < iData.size()-2:
                    yy1 = yMap.transform(iData.interval(i+1).minValue())
                    yy2 = yMap.transform(iData.interval(i+1).maxValue())

                    if y2 == min(yy1, yy2):
                        xx2 = xMap.transform(iData.interval(i+1).minValue())
                        if xx2 != x0 and ((xx2 < x0 and x2 < x0)
                                          or (xx2 > x0 and x2 > x0)):
                            # One pixel distance between neighboured bars
                            y2 += 1

                self.drawBar(
                    painter, Qt.Horizontal, QRect(x0, y1, x2-x0, y2-y1))
            else:
                y2 = yMap.transform(iData.value(i))
                if y2 == y0:
                    continue

                x1 = xMap.transform(iData.interval(i).minValue())
                x2 = xMap.transform(iData.interval(i).maxValue())

                if x1 > x2:
                    x1, x2 = x2, x1

                if i < iData.size()-2:
                    xx1 = xMap.transform(iData.interval(i+1).minValue())
                    xx2 = xMap.transform(iData.interval(i+1).maxValue())
                    x2 = min(xx1, xx2)
                    yy2 = yMap.transform(iData.value(i+1))
                    if x2 == min(xx1, xx2):
                        if yy2 != 0 and (( yy2 < y0 and y2 < y0)
                                         or (yy2 > y0 and y2 > y0)):
                            # One pixel distance between neighboured bars
                            x2 -= 1
                
                self.drawBar(
                    painter, Qt.Vertical, QRect(x1, y0, x2-x1, y2-y0))

    def setBaseline(self, reference):
        if self.baseline() != reference:
            self.__reference = reference
            self.itemChanged()

    def baseline(self,):
        return self.__reference

    def setHistogramAttribute(self, attribute, on = True):
        if self.testHistogramAttribute(attribute):
            return

        if on:
            self.__attributes |= attribute
        else:
            self.__attributes &= ~attribute

        self.itemChanged()
    
    def testHistogramAttribute(self, attribute):
        return bool(self.__attributes & attribute) 
           
#=====================================================================
class PlotManager(QtCore.QObject):

    # __pyqtSignals__ = ("newPlotWindow(QtCore.QMutex)",)
    # @QtCore.pyqtSignature("newPlotWindow(QtCore.QMutex)")
    
    # def emitNewPlotWindow(self,_mutex):
        # self.emit(SIGNAL("newPlotWindow(QtCore.QMutex)") , _mutex)
    
    newPlotWindowSignal = QtCore.pyqtSignal( (QtCore.QMutex, ))

    
    def __init__(self,_viewManager=None,_plotSupportFlag=False):
        QtCore.QObject.__init__(self, None)
        self.vm = _viewManager
        print MODULENAME,' class PlotManager():  __init__():  self.vm=',self.vm
        self.plotsSupported = _plotSupportFlag
        self.plotWindowList = []
        self.plotWindowMutex = QtCore.QMutex()
        self.signalsInitialized = False
                     
    def getPlotWindow(self):
        if self.plotsSupported:
           return PlotWindow()
        else:
            return PlotWindowBase()
            
    def reset(self):
        self.plotWindowList=[]
        
        
    def initSignalAndSlots(self):
        # since initSignalAndSlots can be called in SimTabView multiple times (after each simulation restart) we have to ensure that signals are connected only once 
        # otherwise there will be an avalanche of signals - each signal for each additional simulation run this will cause lots of extra windows to pop up 
        if not self.signalsInitialized: 
            self.newPlotWindowSignal.connect(self.processRequestForNewPlotWindow)
            self.signalsInitialized=True
        # self.connect(self,SIGNAL("newPlotWindow(QtCore.QMutex)"),self.processRequestForNewPlotWindow)
        

    def getNewPlotWindow(self):   # rf. SimpleTabView.py: addNewGraphicsWindow()
        print MODULENAME,"  getNewPlotWindow() "
        self.plotWindowMutex.lock()

        self.newPlotWindowSignal.emit(self.plotWindowMutex)   # calls "processRequestForNewPlotWindow"
        # processRequestForNewPlotWindow will be called and it will unlock drawMutex but before it will finish runnning (i.e. before the new window is actually added)we must make sure that getNewPlotwindow does not return 
        self.plotWindowMutex.lock()
        self.plotWindowMutex.unlock()
        print MODULENAME,"  getNewPlotWindow():  returning  ",self.plotWindowList[-1]
        print MODULENAME,"  getNewPlotWindow():  dir =",dir(self.plotWindowList[-1])
        return self.plotWindowList[-1] # returning recently added window
        

    def processRequestForNewPlotWindow(self,_mutex):
        print MODULENAME,"processRequestForNewPlotWindow():  self.vm.useVTKPlots=",self.vm.useVTKPlots
        if not self.vm.useVTKPlots:
            from  Graphics.PlotFrameWidget import PlotFrameWidget
        else:
            from  Graphics.PlotFrameWidgetVTK import PlotFrameWidget
            
        print MODULENAME,"processRequestForNewPlotWindow():  mutex=",_mutex
        if not self.plotsSupported:
            return PlotWindowInterfaceBase(None) # dummy PlotwindowInterface
        
#        from  Graphics.PlotFrameWidget import PlotFrameWidget
        if not self.vm.simulationIsRunning:
            return
        # self.vm.simulation.drawMutex.lock()
        
        self.vm.windowCounter += 1        
        newWindow = PlotFrameWidget(self.vm)   # <rwh-----------------------  (in /Graphics)
#        newWindow = GraphicsFrameWidget(self.vm)   # <rwh-----------------------  (in /Graphics)
        print MODULENAME,"processRequestForNewPlotWindow():  newWindow=",newWindow
        
        self.vm.windowDict[self.vm.windowCounter] = newWindow
        self.vm.plotWindowDict[self.vm.windowCounter] = self.vm.windowDict[self.vm.windowCounter]
        
        newWindow.setWindowTitle("Plot Window "+ str(self.vm.windowCounter))
        
        self.vm.lastActiveWindow = newWindow
        # # self.updateWindowMenu()
          
        newWindow.setShown(False)
        
        self.vm.mdiWindowDict[self.vm.windowCounter] = self.vm.addSubWindow(newWindow)
        newWindow.show()
        
        plotWindowInterface = PlotWindowInterface(newWindow)   # <rwh----------------------
        self.plotWindowList.append(plotWindowInterface) # store plot window interface in the window list
        
        self.plotWindowMutex.unlock()

