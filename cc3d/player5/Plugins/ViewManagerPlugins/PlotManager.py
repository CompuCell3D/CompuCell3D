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

    #
    # def getPlotWindowsLayoutDict(self):
    #     windowsLayout = {}
    #     from Graphics.GraphicsWindowData import GraphicsWindowData
    #
    #
    #     for plotInterface in self.plotWindowList:
    #         gwd = GraphicsWindowData()
    #         gwd.sceneName =  plotInterface.title
    #         gwd.winType =  'plot'
    #         plotWindow = plotInterface.plotWindow
    #         mdiPlotWindow = self.vm.findMDISubWindowForWidget(plotWindow)
    #         print 'plotWindow=',plotWindow
    #         print 'mdiPlotWindow=',mdiPlotWindow
    #         gwd.winSize = mdiPlotWindow.size()
    #         gwd.winPosition = mdiPlotWindow.pos()
    #
    #         windowsLayout[gwd.sceneName] = gwd.toDict()
    #
    #     return windowsLayout

    def processRequestForNewPlotWindow(self, _mutex, obj):
        print('obj=', obj)
        #        print MODULENAME,"processRequestForNewPlotWindow(): GOT HERE mutex=",_mutex
        if not self.plotsSupported:
            return PlotWindowInterfaceBase(None)  # dummy PlotwindowInterface


        if not self.vm.simulationIsRunning:
            return
        # self.vm.simulation.drawMutex.lock()

        # MDIFIX
        # self.vm.windowCounter += 1

        newWindow = PlotFrameWidget(self.vm, **obj)

        # newWindow.resizePlot(600,600)

        # MDIFIX
        # self.vm.windowDict[self.vm.windowCounter]=newWindow
        # self.vm.plotWindowDict[self.vm.windowCounter] = self.vm.windowDict[self.vm.windowCounter]

        # newWindow.setWindowTitle("Plot Window "+str(self.vm.windowCounter))
        # MDIFIX
        # self.vm.lastActiveWindow = newWindow

        # # self.updateWindowMenu()

        #

        newWindow.show()

        mdiPlotWindow = self.vm.addSubWindow(newWindow)

        mdiPlotWindow.setWindowTitle(obj['title'])

        suggested_win_pos = self.vm.suggested_window_position()

        if suggested_win_pos.x() != -1 and suggested_win_pos.y() != -1:
            mdiPlotWindow.move(suggested_win_pos)

        self.vm.lastActiveRealWindow = mdiPlotWindow

        # mdiPlotWindow = self.vm.lastActiveWindow

        # newWindow.setFixedSize(600,600)
        # newWindow.resize(600,600)
        # newWindow.resizePlot(600,600)
        # # # newWindow.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding))
        # # # # newWindow.adjustSize()
        # # # newWindow.showMinimized()
        # # # newWindow.showNormal()
        # newWindow.resize(600,600)

        # newWindow.resizePlot(600,600)


        # def resizeHandler(ev):
        # print 'RESIZE HANDLER'
        # print 'ev.oldSize() =',ev.oldSize()
        # print 'ev.size() =',ev.size()

        # import time
        # time.sleep(2)

        # mdiPlotWindow.resizeEvent = resizeHandler
        print('mdiPlotWindow=', mdiPlotWindow)
        print('newWindow=', newWindow)
        # mdiPlotWindow.setFixedSize(600,600)
        # mdiPlotWindow.resize(300, 300)
        # mdiPlotWindow.widget().resize(600,600)

        # import time
        # time.sleep(2)

        # mdiPlotWindow.resize(400,400)
        # MDIFIX
        # self.vm.mdiWindowDict[self.vm.windowCounter] = mdiPlotWindow

        # self.vm.mdiWindowDict[self.vm.windowCounter]=self.vm.addSubWindow(newWindow)
        newWindow.show()

        plotWindowInterface = PlotWindowInterface(newWindow)
        self.plotWindowList.append(plotWindowInterface)  # store plot window interface in the window list

        # self.restoreSingleWindow(plotWindowInterface)

        self.plotWindowMutex.unlock()

        # return  plotWindowInterface# here I will need to call either PlotWindowInterface or PlotWindowInterfaceBase depending if dependencies are installed or not
        # def updatePlots(self):
        # for plotWindowInterface in self.plotWindowList:
        # plotWindowInterface.showAllPlots()

        # def addNewPlotWindow(self):
        # print "\n\n\n\n GOT HERE"
        # if not self.plotsSupported:
        # return PlotWindowInterfaceBase(None) # dummy PlotwindowInterface

        # from  Graphics.PlotFrameWidget import PlotFrameWidget
        # if not self.vm.simulationIsRunning:
        # return
        # self.vm.simulation.drawMutex.lock()

        # self.vm.windowCounter+=1
        # newWindow=PlotFrameWidget(self.vm)

        # self.vm.windowDict[self.vm.windowCounter]=newWindow

        # newWindow.setWindowTitle("Plot Window "+str(self.vm.windowCounter))




        # self.vm.lastActiveWindow=newWindow
        # # # self.updateWindowMenu()


        # newWindow.setShown(False)

        # self.vm.addSubWindow(newWindow)
        # newWindow.show()

        # plotWindowInterface=PlotWindowInterface(newWindow)
        # self.plotWindowList.append(plotWindowInterface) # store plot window interface in the window list


        # self.vm.simulation.drawMutex.unlock()
        # return  plotWindowInterface# here I will need to call either PlotWindowInterface or PlotWindowInterfaceBase depending if dependencies are installed or not

        # //////////////////////////////////////////////////////////////////////

    # class CustomPlot:
    # def __init__(self,_plotManager):
    # self.pM=_plotManager
    # print "self.pM=",self.pM," function list=",dir(self.pM)

    # self.pW=self.pM.addNewPlotWindow()

    # def initPlot(self):
    # self.pW.setTitle("Custom Plot")
    # self.pW.setXAxisTitle("MonteCarlo Step (MCS)")
    # self.pW.setYAxisTitle("Variables")
    # self.pW.addGrid()
    # # plot 1
    # self.pW.addPlot("MCS")
    # self.pW.addDataPoint("MCS",1,1)
    # self.pW.addDataPoint("MCS",2,2)
    # self.pW.changePlotProperty("MCS","LineWidth",5)
    # self.pW.changePlotProperty("MCS","LineColor","red")
    # # self.pW.showPlot("MCS")

    # # plot 1
    # self.pW.addPlot("MCS1")
    # self.pW.addDataPoint("MCS1",1,-1)
    # self.pW.addDataPoint("MCS1",2,2)
    # self.pW.changePlotProperty("MCS1","LineWidth",1)
    # self.pW.changePlotProperty("MCS1","LineColor","green")
    # # self.pW.showPlot("MCS1")
    # self.pW.showAllPlots()

# //////////////////////////////////////////////////////////////////////

#
# MODULENAME = '---- PlotManager.py: '
#
# PLOT_TYPE_POSITION=3
# (XYPLOT,HISTOGRAM,BARPLOT)=range(0,3)
# MAX_FIELD_LEGTH=25
#
# # Notice histogram and Bar Plot implementations need more work. They are functional but have a bit strange syntax and for Bar Plot we can only plot one series per plot
#
# # class PlotWindowInterface(PlotManagerSetup.PlotWindowInterfaceBase,QtCore.QObject):
# class PlotWindowInterface(QtCore.QObject):
#     # showPlotSignal = QtCore.pyqtSignal( (QtCore.QString,QtCore.QMutex, ))
#     # showBarCurvePlotSignal = QtCore.pyqtSignal( ('char*',QtCore.QMutex, ))
#     # savePlotAsPNGSignal=QtCore.pyqtSignal( ('char*','int','int',QtCore.QMutex, )) #savePlotAsPNG has to emit signal with locking mutex to work correctly
#     # showPlotSignal = QtCore.pyqtSignal( ('char*',QtCore.QMutex, ))
#     # showHistPlotSignal = QtCore.pyqtSignal( ('char*',QtCore.QMutex, ))
#     # showPlotSignal = QtCore.pyqtSignal( (QtCore.QString,QtCore.QMutex, ))
#     # showAllPlotsSignal=QtCore.pyqtSignal( (QtCore.QMutex, ))
#     # showHistPlotSignal = QtCore.pyqtSignal( (QtCore.QString,QtCore.QMutex, ))
#     # showAllHistPlotsSignal=QtCore.pyqtSignal( (QtCore.QMutex, ))
#     #
#     # showBarCurvePlotSignal = QtCore.pyqtSignal( (QtCore.QString,QtCore.QMutex, ))
#     # showAllBarCurvePlotsSignal=QtCore.pyqtSignal( (QtCore.QMutex, ))
#     #
#     # savePlotAsPNGSignal=QtCore.pyqtSignal( (QtCore.QString,'int','int',QtCore.QMutex, )) #savePlotAsPNG has to emit signal with locking mutex to work correctly
#
#
#     #IMPORTANT: turns out that using QString in signals is essential to get correct behavior. for some reason using char * in signal may malfunction e.g. wrong string is sent to slot.
#
#     showPlotSignal = QtCore.pyqtSignal(str, QtCore.QMutex)
#     showAllPlotsSignal = QtCore.pyqtSignal(QtCore.QMutex)
#     showHistPlotSignal = QtCore.pyqtSignal(str, QtCore.QMutex)
#     showAllHistPlotsSignal = QtCore.pyqtSignal(QtCore.QMutex)
#
#     showBarCurvePlotSignal = QtCore.pyqtSignal(str,QtCore.QMutex)
#     showAllBarCurvePlotsSignal = QtCore.pyqtSignal(QtCore.QMutex)
#
#     # savePlotAsPNG has to emit signal with locking mutex to work correctly
#     savePlotAsPNGSignal = QtCore.pyqtSignal(str, int, int, QtCore.QMutex)
#
#
#
#     def __init__(self, _plotWindow=None):
#         # PlotManagerSetup.PlotWindowInterfaceBase.__init__(self,_plotWindow)
#         QtCore.QObject.__init__(self, None)
#         if _plotWindow:
#             self.plotWindow = _plotWindow
#             import weakref
#             self.plotWindow.plotInterface = weakref.ref(self)
#             self.pW = self.plotWindow.plotWidget
#
#         self.plotData = {}
#         self.plotHistData = {}
#         self.plotDrawingObjects = {}
#         self.initSignalsAndSlots()
#         self.plotWindowInterfaceMutex = QtCore.QMutex()
#         self.dirtyFlagIndex = 2 # this is the index of the flag tha is used to signal wheather the data has been modified or not
#         self.autoLegendFlag = False
#         self.legendSetFlag = False
#         # self.legendPosition = Qwt.QwtPlot.BottomLegend
#         #todo
#         self.legendPosition = None
#
#
#         self.barplot = None
#
#         self.eraseAllFlag = False
#         self.logScaleFlag = False
#         self.title = ''
#
#     def getQWTPLotWidget(self): # returns native QWT widget to be manipulated by expert users
#         return self.plotWindow
#
#     def initSignalsAndSlots(self):
#         self.showAllPlotsSignal.connect(self.__showAllPlots)
#         self.showPlotSignal.connect(self.__showPlot)
#         self.showAllHistPlotsSignal.connect(self.__showAllHistPlots)
#         self.showHistPlotSignal.connect(self.__showHistPlot)
#         self.showAllBarCurvePlotsSignal.connect(self.__showAllBarCurvePlots)
#         self.showBarCurvePlotSignal.connect(self.__showBarCurvePlot)
#         self.savePlotAsPNGSignal.connect(self.__savePlotAsPNG)
#
#     def clear(self):
#         # self.pW.clear()
#         self.pW.detachItems()
#
#     def replot(self):
#         self.pW.replot()
#
#     def setTitle(self,_title):
#         self.title=str(_title)
#         self.pW.setTitle(_title)
#
#     def setTitleSize(self,_size):
#         title=self.pW.title()
#         font=title.font()
#         font.setPointSize(_size)
#         title.setFont(font)
#         self.pW.setTitle(title)
#
#     def setTitleColor(self,_colorName):
#         title=self.pW.title()
#         title.setColor(QColor(_colorName))
#         self.pW.setTitle(title)
#
#     def setPlotBackgroundColor(self,_colorName):
#         self.pW.setCanvasBackground(QColor(_colorName))
#
#     def addAutoLegend(self,_position="bottom"):
#         self.autoLegendFlag = True
#
#         # print "_position=",_position
#         # sys.exit()
#
#         if _position.lower() == "top":
#             self.legendPosition=Qwt.QwtPlot.TopLegend
#             # print "_position=",_position
#             # sys.exit()
#
#         elif _position.lower() == "bottom":
#             self.legendPosition = Qwt.QwtPlot.BottomLegend
#         elif _position.lower() == "left":
#             self.legendPosition = Qwt.QwtPlot.LeftLegend
#         elif _position.lower() == "right":
#             self.legendPosition = Qwt.QwtPlot.RightLegend
#
#
#         # self.legend = Qwt.QwtLegend()
#         # self.legend.setFrameStyle(QFrame.Box | QFrame.Sunken)
#         # self.legend.setItemMode(Qwt.QwtLegend.ClickableItem)
#
#         # self.pW.insertLegend(legend, Qwt.QwtPlot.TopLegend)
#
#     def addPlot(self, _plotName, _style="Lines", _color='black', _size=1):
#         # self.plotData[_plotName]=[arange(0),arange(0),False]
#
#         self.plotData[_plotName]=[array([],dtype=double),array([],dtype=double),False,XYPLOT]
#
#         self.plotDrawingObjects[_plotName] = {"curve":Qwt.QwtPlotCurve(_plotName),"LineWidth":_size,"LineColor":_color}
#         plotStyle=getattr(Qwt.QwtPlotCurve,_style)
#         # self.plotDrawingObjects[_plotName]["curve"].setStyle(Qwt.QwtPlotCurve.Dots)
#         self.plotDrawingObjects[_plotName]["curve"].setStyle(plotStyle)
#
#     def addGrid(self):
#         grid = Qwt.QwtPlotGrid()
#         grid.attach(self.pW)
#         grid.setPen(QPen(Qt.black, 0, Qt.DotLine))
#
#     def eraseAllData(self):
#
#         self.cleanAllContainers()
#
#         for name, data in self.plotData.iteritems():
#
#             data[self.dirtyFlagIndex]=True
#
#         self.eraseAllFlag=True
#
#     def cleanAllContainers(self):
#
#         for name,data in self.plotData.iteritems():
#
#             data[0].resize(0)
#             data[1].resize(0)
#             data[self.dirtyFlagIndex]=True
#
#     def eraseData(self,_plotName):
#         plotType=self.plotData[_plotName][PLOT_TYPE_POSITION]
#         self.plotData[_plotName]=[array([],dtype=double),array([],dtype=double),False,plotType]
#
#     def addDataPoint(self,_plotName, _x,_y):
#
#         if not _plotName in self.plotData.keys():
#
#             return
#
#         if self.eraseAllFlag:
#
#             self.cleanAllContainers()
#             self.eraseAllFlag=False
#
#         currentLength=len(self.plotData[_plotName][0])
#         self.plotData[_plotName][0].resize(currentLength+1)
#         self.plotData[_plotName][1].resize(currentLength+1)
#
#         self.plotData[_plotName][0][currentLength]=_x
#         self.plotData[_plotName][1][currentLength]=_y
#         self.plotData[_plotName][self.dirtyFlagIndex]=True
#         # print "self.plotData[_plotName][0]=",self.plotData[_plotName][0]
#         # print "self.plotData[_plotName][1]=",self.plotData[_plotName][1]
#
#
#     def getDrawingObjectsSettings(self,_plotName):
#         if _plotName in self.plotDrawingObjects.keys():
#             return self.plotDrawingObjects[_plotName]
#         else:
#             return None
#
#     def changePlotProperty(self,_plotName,_property,_value):
#         self.plotDrawingObjects[_plotName][_property]=_value
#
#     def setXAxisTitle(self,_title):
#         self.pW.setAxisTitle(Qwt.QwtPlot.xBottom, _title)
#
#     def setYAxisTitle(self,_title):
#         self.pW.setAxisTitle(Qwt.QwtPlot.yLeft, _title)
#
#     def setXAxisTitleSize(self,_size):
#
#         title=self.pW.axisTitle(Qwt.QwtPlot.xBottom)
#         font=title.font()
#         font.setPointSize(_size)
#         title.setFont(font)
#         self.pW.setAxisTitle(Qwt.QwtPlot.xBottom, title)
#
#     def setXAxisTitleColor(self,_colorName):
#
#         title=self.pW.axisTitle(Qwt.QwtPlot.xBottom)
#         title.setColor(QColor(_colorName))
#         self.pW.setAxisTitle(Qwt.QwtPlot.xBottom, title)
#
#     def setYAxisTitleSize(self,_size):
#
#         title=self.pW.axisTitle(Qwt.QwtPlot.yLeft)
#         font=title.font()
#         font.setPointSize(_size)
#         title.setFont(font)
#         self.pW.setAxisTitle(Qwt.QwtPlot.yLeft, title)
#
#     def setYAxisTitleColor(self,_colorName):
#
#         title=self.pW.axisTitle(Qwt.QwtPlot.yLeft)
#         title.setColor(QColor(_colorName))
#         self.pW.setAxisTitle(Qwt.QwtPlot.yLeft, title)
#
#
#     def setXAxisLogScale(self):
#         self.pW.setAxisScaleEngine(Qwt.QwtPlot.xBottom, Qwt.QwtLog10ScaleEngine())
#         self.logScaleFlag=True
#
#     def setYAxisLogScale(self):
#         self.pW.setAxisScaleEngine(Qwt.QwtPlot.yLeft, Qwt.QwtLog10ScaleEngine())
#         self.logScaleFlag=True
#
#     def setYAxisScale(self,_lower=0.0,_upper=100.0):
#         self.pW.setAxisScale(Qwt.QwtPlot.yLeft, _lower, _upper)
#
#     def setXAxisScale(self,_lower=0.0,_upper=100.0):
#         self.pW.setAxisScale(Qwt.QwtPlot.xBottom, _lower, _upper)
#
#
#     def showPlot(self,_plotName):
#         self.plotWindowInterfaceMutex.lock()
#         self.showPlotSignal.emit(QString(_plotName),self.plotWindowInterfaceMutex)
#
#     def  savePlotAsPNG(self,_fileName,_sizeX=400,_sizeY=400) :
#         self.plotWindowInterfaceMutex.lock()
#         self.savePlotAsPNGSignal.emit(_fileName,_sizeX,_sizeY,self.plotWindowInterfaceMutex)
#
#     def __savePlotAsPNG(self,_fileName,_sizeX,_sizeY,_mutex):
#         fileName=str(_fileName)
# #        pixmap=QPixmap(_sizeX,_sizeY)  # worked on Windows, but not Linux/OSX
# #        pixmap.fill(QColor("white"))
#
#         imgmap = QImage(_sizeX, _sizeY, QImage.Format_ARGB32)
#         #imgmap.fill(Qt.white)
#         imgmap.fill(qRgba(255, 255, 255, 255)) # solid white background (should probably depend on user-chosen colors though)
#
#         self.pW.print_(imgmap)
#         # following seems pretty crude, but keep in mind user can change Prefs anytime during sim
# # # #         if Configuration.getSetting("OutputToProjectOn"):
# # # #             outDir = str(Configuration.getSetting("ProjectLocation"))
# # # #         else:
# # # #             outDir = str(Configuration.getSetting("OutputLocation"))
#
#         import CompuCellSetup
#         outDir=CompuCellSetup.getSimulationOutputDir()
#
#         outfile = os.path.join(outDir,fileName)
# #        print '--------- savePlotAsPNG: outfile=',outfile
#         imgmap.save(outfile,"PNG")
#         _mutex.unlock()
#
#     #original implementation - does not really work unless we use signal slot mechanism
#     # def savePlotAsPNG(self,_fileName,_sizeX=400,_sizeY=400):
# # #        pixmap=QPixmap(_sizeX,_sizeY)  # worked on Windows, but not Linux/OSX
# # #        pixmap.fill(QColor("white"))
#
#         # imgmap = QImage(_sizeX, _sizeY, QImage.Format_ARGB32)
#         # #imgmap.fill(Qt.white)
#         # imgmap.fill(qRgba(255, 255, 255, 255)) # solid white background (should probably depend on user-chosen colors though)
#
#         # self.pW.print_(imgmap)
#         # # following seems pretty crude, but keep in mind user can change Prefs anytime during sim
#         # if Configuration.getSetting("OutputToProjectOn"):
#             # outDir = str(Configuration.getSetting("ProjectLocation"))
#         # else:
#             # outDir = str(Configuration.getSetting("OutputLocation"))
#         # outfile = os.path.join(outDir,_fileName)
# # #        print '--------- savePlotAsPNG: outfile=',outfile
#         # imgmap.save(outfile,"PNG")
#
#     def writeOutHeader(self, _file,_plotName, _outputFormat=LEGACY_FORMAT):
#
#         if _outputFormat==LEGACY_FORMAT:
#
#             _file.write(_plotName+ '\n')
#             return 0 # field width
#
#         elif _outputFormat==CSV_FORMAT:
#
#             plotName = _plotName.replace(' ' , '_')
#
#             fieldSize = len(plotName)+2 # +2 is for _x or _y
#             if MAX_FIELD_LEGTH > fieldSize:
#                 fieldSize=MAX_FIELD_LEGTH
#
#             fmt=''
#             fmt+='{0:>'+str(fieldSize)+'},'
#             fmt+='{1:>'+str(fieldSize)+'}\n'
#
#             _file.write(fmt.format(plotName+'_x',plotName+'_y') )
#
#             return fieldSize
#
#         else:
#             raise LookupError(MODULENAME+" writeOutHeader :"+"Requested output format: "+outputFormat+" does not exist")
#
#
#     def savePlotAsData(self,_fileName,_outputFormat=LEGACY_FORMAT):
#         # PLOT_TYPE_POSITION=3
# # (XYPLOT,HISTOGRAM,BARPLOT)=range(0,3)
#
#         import CompuCellSetup
#         outDir=CompuCellSetup.getSimulationOutputDir()
#
#         outfile = os.path.join(outDir,_fileName)
#         # print MODULENAME,'  savePlotAsData():   outfile=',outfile
#         fpout = open(outfile, "w")
#         # print MODULENAME,'  self.plotData= ',self.plotData
#
#         for plotName,plotData  in self.plotData.iteritems():
#             # fpout.write(plotName+ '\n')
#             fieldSize=self.writeOutHeader( _file=fpout,_plotName=plotName, _outputFormat=_outputFormat)
#
#             xvals = plotData[0]
#             yvals = plotData[1]
#             # print MODULENAME,'  savePlotAsData():   xvals=',xvals
#             # print MODULENAME,'  savePlotAsData():   yvals=',yvals
#             if _outputFormat==LEGACY_FORMAT:
#                 if plotData[PLOT_TYPE_POSITION]==XYPLOT or plotData[PLOT_TYPE_POSITION]==BARPLOT:
#                     for jdx in range(len(xvals)):
#                         xyStr = "%f  %f\n" % (xvals[jdx],yvals[jdx])
#                         fpout.write(xyStr)
#                 elif plotData[PLOT_TYPE_POSITION]==HISTOGRAM:
#                     for jdx in range(len(xvals)-1):
#                         xyStr = "%f  %f\n" % (xvals[jdx],yvals[jdx])
#                         fpout.write(xyStr)
#
#             elif  _outputFormat==CSV_FORMAT:
#                 fmt=''
#                 fmt+='{0:>'+str(fieldSize)+'},'
#                 fmt+='{1:>'+str(fieldSize)+'}\n'
#
#                 if plotData[PLOT_TYPE_POSITION]==XYPLOT or plotData[PLOT_TYPE_POSITION]==BARPLOT:
#                     for jdx in range(len(xvals)):
#
#                         xyStr = fmt.format(xvals[jdx],yvals[jdx])
#                         # "%f  %f\n" % (xvals[jdx],yvals[jdx])
#                         fpout.write(xyStr)
#                 elif plotData[PLOT_TYPE_POSITION]==HISTOGRAM:
#                     for jdx in range(len(xvals)-1):
#                         xyStr = fmt.format(xvals[jdx],yvals[jdx])
#                         # xyStr = "%f  %f\n" % (xvals[jdx],yvals[jdx])
#                         fpout.write(xyStr)
#
#
#             else:
#                 raise LookupError(MODULENAME+" savePlotAsData :"+"Requested output format: "+outputFormat+" does not exist")
#             fpout.write('\n') #separating data series by a line
#
#         fpout.close()
#
#     def __showPlot(self,plotName,_mutex=None):
#         _plotName=str(plotName)
#         if (not _plotName in self.plotData.keys() ) or (not _plotName in self.plotDrawingObjects.keys() ):
#             return
#         if not self.plotData[_plotName][self.dirtyFlagIndex]:
#             return
#         drawingObjects=self.plotDrawingObjects[_plotName]
#         drawingObjects["curve"].attach(self.pW)
#         drawingObjects["curve"].setPen(QPen(QColor(drawingObjects["LineColor"]), drawingObjects["LineWidth"]))
#         drawingObjects["curve"].setData(self.plotData[_plotName][0], self.plotData[_plotName][1])
#         self.plotData[_plotName][self.dirtyFlagIndex]=False
#         if self.autoLegendFlag and not self.legendSetFlag:
#             self.legend = Qwt.QwtLegend()
#             self.legend.setFrameStyle(QFrame.Box | QFrame.Sunken)
#             self.legend.setItemMode(Qwt.QwtLegend.ClickableItem)
#             self.pW.insertLegend(self.legend, self.legendPosition)
#             self.legendSetFlag=True
#         self.pW.replot()
#         self.plotWindowInterfaceMutex.unlock()
#
#
#     def showAllPlots(self):
#         self.plotWindowInterfaceMutex.lock()
#         self.showAllPlotsSignal.emit(self.plotWindowInterfaceMutex)
#
#     def __showAllPlots(self,_mutex=None):
#
#         for plotName in self.plotData.keys():
#             if self.plotData[plotName][self.dirtyFlagIndex]:
#                 if plotName in self.plotDrawingObjects.keys():
#                     drawingObjects=self.plotDrawingObjects[plotName]
#                     drawingObjects["curve"].attach(self.pW)
#                     drawingObjects["curve"].setPen(QPen(QColor(drawingObjects["LineColor"]), drawingObjects["LineWidth"]))
#                     drawingObjects["curve"].setData(self.plotData[plotName][0], self.plotData[plotName][1])
#
# #                    print "self.legendPosition=",self.legendPosition
#                     if self.autoLegendFlag and not self.legendSetFlag:
#                         self.legend = Qwt.QwtLegend()
#                         self.legend.setFrameStyle(QFrame.Box | QFrame.Sunken)
#                         self.legend.setItemMode(Qwt.QwtLegend.ClickableItem)
#                         self.pW.insertLegend(self.legend, self.legendPosition)
#                         self.legendSetFlag=True
#
#                     self.pW.replot()
#                     self.plotData[plotName][self.dirtyFlagIndex]=False
#         _mutex.unlock()
#
#     def showAllHistPlots(self):
#         self.plotWindowInterfaceMutex.lock()
#         self.showAllHistPlotsSignal.emit(self.plotWindowInterfaceMutex)
#
#     def __showHistPlot(self,plotName,_mutex=None):
#       _plotName=str(plotName)
#       print _plotName
#       self.histogram.attach(self.pW)
#       self.pW.replot()
#       self.plotWindowInterfaceMutex.unlock()
#
#     def __showAllHistPlots(self,_mutex=None):
#         for hist in self.plotHistData.values():
#             hist.attach(self.pW)
#         self.pW.replot()
#         _mutex.unlock()
#
#     def addHistogram(self, plot_name , value_array ,  number_of_bins):
#         import numpy
#         (values, intervals) = numpy.histogram(value_array, bins=number_of_bins)
#         self.addHistPlotData(_plotName=plot_name, _values=values, _intervals=intervals)
#
#     def addHistPlotData(self,_plotName,_values,_intervals):
#         # print 'addHistPlotData'
#         # print '_values=',_values
#         # print '_intervals=',_intervals
#         # self.plotData[_plotName]=[array([],dtype=double),array([],dtype=double),False]
#
#         self.plotData[str(_plotName)]=[_intervals,_values,False,HISTOGRAM]
#
#         intervals = []
#         valLength = len(_values)
#         values = Qwt.QwtArrayDouble(valLength)
#         for i in range(valLength):
#         #width = _intervals[i+1]-_intervals[i]+2
#             intervals.append(Qwt.QwtDoubleInterval(_intervals[i], _intervals[i+1])); #numpy automcatically adds extra element for edge
#             values[i] = _values[i]
#
#         self.plotHistData[_plotName].setData(Qwt.QwtIntervalData(intervals, values))
#
#     def addHistPlot(self,_plotName, _r = 100, _g = 100, _b = 0,_alpha=255):
#         self.plotHistData[_plotName]=HistogramItem()
#         self.plotHistData[_plotName].setColor(QColor(_r,_g,_b,_alpha))
#
#     def addHistogramPlot(self,_plotName, _color='black',_alpha=255):
#         self.plotHistData[_plotName]=HistogramItem()
#         color=QColor(_color)
#         color.setAlpha(_alpha)
#         self.plotHistData[_plotName].setColor(color)
#
#     #def setHistogramColor(self,):
#         #self.histogram.setColor(QColor(_colorName))
#
#     def setHistogramColor(self,_colorName = None, _r = 100, _g = 100, _b = 0,_alpha=255):
#         if _colorName != None:
#             # self.histogram.setColor(QColor(_colorName))
#             self.plotHistData[_plotName].setColor(QColor(_colorName))
#         else:
#             # self.histogram.setColor(QColor(_r,_g,_b,_alpha))
#             self.plotHistData[_plotName].setColor(QColor(_r,_g,_b,_alpha))
#
#     def setHistogramView(self):
#         self.histogram = HistogramItem()
#         self.histogram.setColor(Qt.darkCyan)
#
#         numValues = 20
#         intervals = []
#         values = Qwt.QwtArrayDouble(numValues)
#
#         pos = 0.0
#         for i in range(numValues):
#             width = 5 + random.randint(0, 4)
#             value = random.randint(0, 99)
#             intervals.append(Qwt.QwtDoubleInterval(pos, pos+width));
#             values[i] = value
#             pos += width
#
#       #self.histogram.setData(Qwt.QwtIntervalData(intervals, values))
#
#     def showHistogram(self):
#         self.plotWindowInterfaceMutex.lock()
#         self.showAllHistPlotsSignal.emit(self.plotWindowInterfaceMutex)
#
#     def addBarPlotData(self,_values,_positions,_width=1):
#
#         self.plotData[self.title]=[_positions,_values,False,BARPLOT]
#
#         for bar in self.pW.itemList():
#             if isinstance(bar, BarCurve):
#                 bar.detach()
#
#         for i in range(len(_values)):
#             self.barplot = BarCurve()
#             self.barplot.attach(self.pW)
#             self.barplot.setData([float(_positions[i]),float( _positions[i]+_width)], [0, float(_values[i])])
#
#     def setBarPlotView(self):
#         #do nothing
#         pass
#
#     def showAllBarCurvePlots(self):
#         self.plotWindowInterfaceMutex.lock()
#         self.showAllBarCurvePlotsSignal.emit(self.plotWindowInterfaceMutex)
#
#     def __showBarCurvePlot(self,_plotName,_mutex=None):
#         plotName=str(_plotName)
#         self.pW.replot()
#         self.plotWindowInterfaceMutex.unlock()
#
#     def __showAllBarCurvePlots(self,_mutex=None):
#         if self.barplot is not None:
#             self.barplot.attach(self.pW)
#         self.pW.replot()
#         _mutex.unlock()
#

# class BarCurve(Qwt.QwtPlotCurve): #courtesy of pyqwt examples
#
#     def __init__(self, penColor=Qt.black, brushColor=Qt.white):
#         Qwt.QwtPlotCurve.__init__(self)
#         self.penColor = penColor
#         self.brushColor = brushColor
#
#     # __init__()
#
#     def drawFromTo(self, painter, xMap, yMap, start, stop):
#         """Draws rectangles with the corners taken from the x- and y-arrays.
#         """
#         painter.setPen(QPen(self.penColor, 2))
#         painter.setBrush(self.brushColor)
#         if stop == -1:
#             stop = self.dataSize()
#         # force 'start' and 'stop' to be even and positive
#         if start & 1:
#             start -= 1
#         if stop & 1:
#             stop -= 1
#         start = max(start, 0)
#         stop = max(stop, 0)
#         for i in range(start, stop, 2):
#             px1 = xMap.transform(self.x(i))
#             py1 = yMap.transform(self.y(i))
#             px2 = xMap.transform(self.x(i+1))
#             py2 = yMap.transform(self.y(i+1))
#             painter.drawRect(px1, py1, (px2 - px1), (py2 - py1))
#
#     # drawFromTo()
#
# # class BarCurve
#
# class HistogramItem(Qwt.QwtPlotItem): #courtesy of pyqwt examples
#
#     Auto = 0
#     Xfy = 1
#
#     def __init__(self, *args):
#         Qwt.QwtPlotItem.__init__(self, *args)
#         self.__attributes = HistogramItem.Auto
#         self.__data = Qwt.QwtIntervalData()
#         self.__color = QColor()
#         self.__reference = 0.0
#         self.setItemAttribute(Qwt.QwtPlotItem.AutoScale, True)
#         self.setItemAttribute(Qwt.QwtPlotItem.Legend, True)
#         self.setZ(20.0)
#
#     # __init__()
#
#     def setData(self, data):
#         self.__data = data
#         self.itemChanged()
#
#     # setData()
#
#     def data(self):
#         return self.__data
#
#     # data()
#
#     def setColor(self, color):
#         if self.__color != color:
#             self.__color = color
#             self.itemChanged()
#
#     # setColor()
#
#     def color(self):
#         return self.__color
#
#     # color()
#
#     def boundingRect(self):
#         result = self.__data.boundingRect()
#         if not result.isValid():
#             return result
#         if self.testHistogramAttribute(HistogramItem.Xfy):
#             result = Qwt.QwtDoubleRect(result.y(), result.x(),
#                                        result.height(), result.width())
#             if result.left() > self.baseline():
#                 result.setLeft(self.baseline())
#             elif result.right() < self.baseline():
#                 result.setRight(self.baseline())
#         else:
#             if result.bottom() < self.baseline():
#                 result.setBottom(self.baseline())
#             elif result.top() > self.baseline():
#                 result.setTop(self.baseline())
#         return result
#
#     # boundingRect()
#
#     #def rtti(self):
#     #print dir(Qwt.QwtPlotItem)
#         #return Qwt.QwtPlotItem.PlotHistogram
#
#     # rtti()
#
#     def draw(self, painter, xMap, yMap, rect):
#         iData = self.data()
#         # # # print iData
#         # # self.xMap=xMap
#         # # self.yMap=yMap
#         # # self.rect=rect
#         # # print 'self.rect=',self.rect
#         # # # print 'xMap=',xMap
#         # # # print 'yMap=',yMap
#
#         painter.setPen(self.color())
#         x0 = xMap.transform(self.baseline())
#         y0 = yMap.transform(self.baseline())
#         # # # print 'self.baseline()=',self.baseline()
#         # # # print 'x0=',x0
#         # # # print 'y0=',y0
#
#         for i in range(iData.size()):
#             if self.testHistogramAttribute(HistogramItem.Xfy):
#                 x2 = xMap.transform(iData.value(i))
#                 if x2 == x0:
#                     continue
#
#                 y1 = yMap.transform(iData.interval(i).minValue())
#                 y2 = yMap.transform(iData.interval(i).maxValue())
#                 # print 'y1=',y1,' y2=',y2
#                 if y1 > y2:
#                     y1, y2 = y2, y1
#
#                 if  i < iData.size()-2:
#                     yy1 = yMap.transform(iData.interval(i+1).minValue())
#                     yy2 = yMap.transform(iData.interval(i+1).maxValue())
#
#                     if y2 == min(yy1, yy2):
#                         xx2 = xMap.transform(iData.interval(i+1).minValue())
#                         if xx2 != x0 and ((xx2 < x0 and x2 < x0)
#                                           or (xx2 > x0 and x2 > x0)):
#                             # One pixel distance between neighboured bars
#                             y2 += 1
#
#                 self.drawBar(
#                     painter, Qt.Horizontal, QRect(x0, y1, x2-x0, y2-y1))
#             else:
#                 y2 = yMap.transform(iData.value(i))
#                 # print 'y2=',y2
#                 if y2 == y0:
#                     continue
#
#                 x1 = xMap.transform(iData.interval(i).minValue())
#                 x2 = xMap.transform(iData.interval(i).maxValue())
#
#                 # # # if x1<x0 or x2<x0:
#                     # # # continue
#
#
#                 if x1 > x2:
#                     x1, x2 = x2, x1
#
#                 if i < iData.size()-2:
#                     xx1 = xMap.transform(iData.interval(i+1).minValue())
#                     xx2 = xMap.transform(iData.interval(i+1).maxValue())
#                     x2 = min(xx1, xx2)
#                     yy2 = yMap.transform(iData.value(i+1))
#                     if x2 == min(xx1, xx2):
#                         if yy2 != 0 and (( yy2 < y0 and y2 < y0)
#                                          or (yy2 > y0 and y2 > y0)):
#                             # One pixel distance between neighboured bars
#                             x2 -= 1
#
#
#                 self.drawBar(painter, Qt.Vertical, QRect(x1, y0, x2-x1, y2-y0))
#
#
#     def setBaseline(self, reference):
#         if self.baseline() != reference:
#             self.__reference = reference
#             self.itemChanged()
#
#     # setBaseLine()
#
#     def baseline(self,):
#         return self.__reference
#
#     # baseline()
#
#     def setHistogramAttribute(self, attribute, on = True):
#         if self.testHistogramAttribute(attribute):
#             return
#
#         if on:
#             self.__attributes |= attribute
#         else:
#             self.__attributes &= ~attribute
#
#         self.itemChanged()
#
#     # setHistogramAttribute()
#
#     def testHistogramAttribute(self, attribute):
#         return bool(self.__attributes & attribute)
#
#     # testHistogramAttribute()
#
#     def drawBar(self, painter, orientation, rect):
#
#         painter.save()
#         color = painter.pen().color()
#         r = rect.normalized()
#         factor = 125;
#         light = color.light(factor)
#         dark = color.dark(factor)
#
#         painter.setBrush(color)
#         painter.setPen(Qt.NoPen)
#         Qwt.QwtPainter.drawRect(painter, r.x()+1, r.y()+1,
#                                 r.width()-2, r.height()-2)
#
#         painter.setBrush(Qt.NoBrush)
#
#         painter.setPen(QPen(light, 2))
#
#         Qwt.QwtPainter.drawLine(painter, r.left()+1, r.top()+2, r.right()+1, r.top()+2)
#
#         painter.setPen(QPen(dark, 2))
#         Qwt.QwtPainter.drawLine(painter, r.left()+1, r.bottom(), r.right()+1, r.bottom())
#
#         painter.setPen(QPen(light, 1))
#         Qwt.QwtPainter.drawLine(painter, r.left(), r.top() + 1, r.left(), r.bottom())
#         Qwt.QwtPainter.drawLine(painter, r.left()+1, r.top()+2, r.left()+1, r.bottom()-1)
#
#         painter.setPen(QPen(dark, 1))
#         Qwt.QwtPainter.drawLine(painter, r.right()+1, r.top()+1, r.right()+1, r.bottom())
#         Qwt.QwtPainter.drawLine(painter, r.right(), r.top()+2, r.right(), r.bottom()-1)
#
#
#         painter.restore()
#
#     # drawBar()

# class HistogramItem
