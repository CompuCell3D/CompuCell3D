class PlotManagerBase:
    def __init__(self,_viewManager=None,_plotSupportFlag=False):
        self.vm=_viewManager
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
    def processRequestForNewPlotWindow(self,_mutex):
        pass

def checkSupportForPlotting():
#    try:
#        import PyQt4.Qwt5            
#        PyQwtFlag=True
#    except ImportError:
#        PyQwtFlag=False
        
    try:
        import numpy            
        numpyFlag=True
    except ImportError:
        numpyFlag=False
    
#    if PyQwtFlag and numpyFlag:
    if numpyFlag:
        return True
    else:
        return False
        
#factory method        
def createPlotManager(_viewManager=None):
    plotSupportFlag=checkSupportForPlotting()
    if plotSupportFlag:
        from PlotManager import PlotManager
        return PlotManager(_viewManager,plotSupportFlag)
    else:
        return PlotManagerBase(_viewManager,plotSupportFlag)
        
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
        
    