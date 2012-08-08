from PySteppables import *
import CompuCell, CompuCellSetup
import time
import sys
import random,numpy
        



            
class HistPlotSteppable(SteppablePy):
    def __init__(self,_simulator,_frequency=10):
        SteppablePy.__init__(self,_frequency)
        self.simulator=_simulator
        self.inventory=self.simulator.getPotts().getCellInventory()
        self.cellList=CellList(self.inventory)

    def start(self):        
        self.pW=CompuCellSetup.viewManager.plotManager.getNewPlotWindow()
        
        if not self.pW:
            return
        
        #initialize setting for Histogram
        
        self.pW.addHistPlot('Hist 1',_r = 0,_g = 255,_b = 0,_alpha = 100)
        self.pW.addHistPlot('Hist 2',_r = 255,_g = 0,_b = 0,_alpha = 255)
        self.pW.addHistPlot('Hist 3',_r = 0,_g = 0,_b = 255,_alpha = 255)
        
        self.pW.setTitle("Histogram of Cell Volumes")
        self.pW.setXAxisTitle("Number of Cells")
        self.pW.setYAxisTitle("Volume Size in Pixels")
        
    def step(self,mcs):
        if not self.pW:
            print "To get scientific plots working you need extra packages installed:"
            print "Windows/OSX Users: Make sure you have numpy installed. For instructions please visit www.compucell3d.org/Downloads"
            print "Linux Users: Make sure you have numpy and PyQwt installed. Please consult your linux distributioun manual pages on how to best install those packages"
            return        
        volList = []
        for cell  in  self.cellList:
            volList.append(cell.volume)
    
    
        gauss = []
        for i in  range(100):
            gauss.append(random.gauss(0,1))
        
        (n2, bins2) = numpy.histogram(gauss, bins=2)  # NumPy version (no plot)
        
        (n3, bins3) = numpy.histogram(volList, bins=50)  # Use NumPy to generate Histogram of volumes
        (n, bins) = numpy.histogram(volList, bins=10)  # Use NumPy to generate Histogram of volumes
        self.pW.addHistPlotData('Hist 2',n2,bins2)
        self.pW.addHistPlotData('Hist 3',n3,bins3)
        self.pW.addHistPlotData('Hist 1',n,bins)
        self.pW.showAllHistPlots()

    
        fileName="HistPlots_"+str(mcs)+".png"
        self.pW.savePlotAsPNG(fileName,1000,1000) # here we specify size of the image saved - default is 400 x 400
    
class BarPlotSteppable(SteppablePy):
    def __init__(self,_simulator,_frequency=10):
        SteppablePy.__init__(self,_frequency)
        self.simulator=_simulator
        self.inventory=self.simulator.getPotts().getCellInventory()
        self.cellList=CellList(self.inventory)

    def start(self):        
        self.pW=CompuCellSetup.viewManager.plotManager.getNewPlotWindow()
        
        if not self.pW:
            return
        
        #initialize setting for BarPlot
        self.pW.setBarPlotView()
        self.pW.setTitle("BarPlot")
        self.pW.setXAxisTitle("Growth of US GDP")
        self.pW.setYAxisTitle("Number of Suits")
        
    def step(self,mcs):
        if not self.pW:
            print "To get scientific plots working you need extra packages installed:"
            print "Windows/OSX Users: Make sure you have numpy installed. For instructions please visit www.compucell3d.org/Downloads"
            print "Linux Users: Make sure you have numpy and PyQwt installed. Please consult your linux distributioun manual pages on how to best install those packages"
            return        
    
        if (mcs%20 == 0):
            gdpList = []
            locations = []
            for i in range(6):
                gdpList.append(random.uniform(1, 100))
                locations.append(random.uniform(1, 20))
            self.pW.addBarPlotData(gdpList,locations,1)
        
        self.pW.showAllBarCurvePlots()
        
    
        fileName="BarPlots_"+str(mcs)+".png"
        self.pW.savePlotAsPNG(fileName,1000,1000) # here we specify size of the image saved - default is 400 x 400
    