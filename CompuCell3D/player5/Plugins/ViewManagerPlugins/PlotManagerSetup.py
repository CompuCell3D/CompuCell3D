class PlotManagerBase:
    def __init__(self, _viewManager=None, _plotSupportFlag=False):
        self.vm = _viewManager
        self.plotsSupported = checkSupportForPlotting()

        # def addNewPlotWindow(self):
        # return PlotWindowInterfaceBase(self.vm)

    def initSignalAndSlots(self):
        pass

    def getPlotWindow(self):
        pass

    def reset(self):
        pass

    def getNewPlotWindow(self):
        pass

    def processRequestForNewPlotWindow(self, _mutex):
        pass

    def restore_plots_layout(self):
        pass

    def getPlotWindowsLayoutDict(self):
        return {}


def checkSupportForPlotting():
    try:
        import pyqtgraph
    except ImportError:
        print 'could not import pyqtgraph'
        return False

    try:
        import numpy
    except ImportError:
        print 'could not import numpy'
        return False

    return True


# def checkSupportForPlotting(_useVTKFlag=False):
#     if _useVTKFlag:   # do we *only* want to use VTK for plotting?
#         PyQwtFlag=False
#     else:
#         try:
#             import PyQt4.Qwt5
#             PyQwtFlag=True
#         except ImportError:
#             PyQwtFlag=False
#
#     vtkPlotFlag=False
#     try:
#         import vtk
#         vtk_version_num = vtk.vtkVersion.GetVTKMajorVersion()*10000+vtk.vtkVersion.GetVTKMinorVersion()*100
#         if vtk_version_num > 50800:
#             vtkPlotFlag = True
#         # if (vtk.vtkVersion.GetVTKMajorVersion() >= 5) and (vtk.vtkVersion.GetVTKMinorVersion() >= 8):  # need to verify this
#         #     vtkPlotFlag=True
#     except ImportError:
#         vtkPlotFlag=False
#
#     try:
#         import numpy
#         numpyFlag=True
#     except ImportError:
#         numpyFlag=False
#
#     if numpyFlag:
#         if PyQwtFlag:
#             return -1
#         elif vtkPlotFlag:
#             return 1
#     else:
#         return 0

# factory method
# def createPlotManager(_viewManager=None, _useVTKFlag=False):   # called from SimpleTabView
#     plotSupportFlag = checkSupportForPlotting(_useVTKFlag)
#     print '------ PlotManagerSetup.py:    plotSupportFlag=',plotSupportFlag
#     if plotSupportFlag < 0:  # via Qwt
#         from PlotManager import PlotManager
#         print '------ PlotManagerSetup.py:    importing PyQwt PlotManager'
#         return PlotManager(_viewManager,plotSupportFlag)
#     elif plotSupportFlag > 0:  # via VTK
#         from PlotManagerVTK import PlotManager
#         return PlotManager(_viewManager,plotSupportFlag)
#     else:  # plotting not possible
#         return PlotManagerBase(_viewManager,plotSupportFlag)

def createPlotManager(_viewManager=None, preferred_manager_type=None):  # called from SimpleTabView
    """

    @param _viewManager:instance of viewManager
    @param preferred_manager_type: not used in currnt API in the future versions
    it will be used to instantiate preferred plotting backend
    @return: instance of PlotManagerBase (inherited)
    """
    # plotSupportFlag = checkSupportForPlotting(_useVTKFlag)


    # import pyqtgraph
    # import PlotManager
    # from PlotManager import PlotManager
    # return PlotManager(_viewManager, True)
    # return None

    try:
        import pyqtgraph
        from PlotManager import PlotManager
        return PlotManager(_viewManager, True)
    except ImportError as e:
        print 'Could not create PLotManager. resorting to PLotManagerBase - plots will not work properly. Here is the exception:'
        print e
        return PlotManagerBase(_viewManager, False)



        # print '------ PlotManagerSetup.py:    plotSupportFlag=', plotSupportFlag
        # if plotSupportFlag < 0:  # via Qwt
        #     from PlotManager import PlotManager
        #     print '------ PlotManagerSetup.py:    importing PyQwt PlotManager'
        #     return PlotManager(_viewManager, plotSupportFlag)
        # elif plotSupportFlag > 0:  # via VTK
        #     from PlotManagerVTK import PlotManager
        #     return PlotManager(_viewManager, plotSupportFlag)
        # else:  # plotting not possible
        #     return PlotManagerBase(_viewManager, plotSupportFlag)
        #



        # def createPlotManager(_viewManager=None, _useVTKFlag=False):  # called from SimpleTabView
    #     plotSupportFlag = checkSupportForPlotting(_useVTKFlag)
    #     print '------ PlotManagerSetup.py:    plotSupportFlag=', plotSupportFlag
    #     if plotSupportFlag < 0:  # via Qwt
    #         from PlotManager import PlotManager
    #         print '------ PlotManagerSetup.py:    importing PyQwt PlotManager'
    #         return PlotManager(_viewManager, plotSupportFlag)
    #     elif plotSupportFlag > 0:  # via VTK
    #         from PlotManagerVTK import PlotManager
    #         return PlotManager(_viewManager, plotSupportFlag)
    #     else:  # plotting not possible
    #         return PlotManagerBase(_viewManager, plotSupportFlag)
    #

    # this class most likel;y is not needed but I keep it for now
    # class PlotWindowInterfaceBase:
    # def __init__(self,_plotWindow=None):
    # if _plotWindow:
    # self.plotWindow=_plotWindow
    # self.pW=self.plotWindow.plotWidget

    # def getQWTPLotWidget(self): # returns native QWT widget to be manipulated byt expert users
    # return self.plotWindow

    # def setPlotTitle(self,_title):
    # pass

    # def setTitleSize(self,_size):
    # pass

    # def setTitleColor(self,_colorName):
    # pass

    # def setPlotBackgroundColor(self,_colorName):
    # pass

    # def addGrid(self):    
    # pass

    # def addPlot(self,_plotName):
    # pass

    # def addDataPoint(self,_plotName, _x,_y):        
    # pass

    # def showPlot(self,_plotName):
    # pass
    # def showAllPlots(self):
    # pass
    # def __showAllPlots(self,_mutex=None):
    # pass
    # def changePlotProperty(self,_plotName,_property,_value):
    # pass

    # def getDrawingObjectsSettings(self,_plotName):
    # return None

    # def setXAxisTitle(self,_title):
    # pass

    # def setXAxisTitleSize(self,_size):
    # pass

    # def setXAxisTitleColor(self,_colorName):
    # pass

    # def setYAxisTitle(self,_title):
    # pass

    # def setYAxisTitleSize(self,_size):
    # pass

    # def setYAxisTitleColor(self,_colorName):
    # pass


    # def setXAxisLogScale(self):
    # pass

    # def setYAxisLogScale(self):
    # pass
