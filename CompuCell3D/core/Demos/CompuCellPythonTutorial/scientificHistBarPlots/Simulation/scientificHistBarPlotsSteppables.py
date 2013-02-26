from PySteppables import *
import CompuCell, CompuCellSetup
import time
import sys
import random,numpy
                    
class HistPlotSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)

    def start(self):        
        
        #initialize setting for Histogram
        self.pW=CompuCellSetup.addNewPlotWindow(_title='Histogram of Cell Volumes',_xAxisTitle='Number of Cells',_yAxisTitle='Volume Size in Pixels')
        self.pW.addHistogramPlot(_plotName='Hist 1',_color='green',_alpha=100)# _alpha is transparency 0 is transparent, 255 is opaque        
        self.pW.addHistogramPlot(_plotName='Hist 2',_color='red')
        self.pW.addHistogramPlot(_plotName='Hist 3',_color='blue')
        
    def step(self,mcs):
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
        fileName="HistPlots_"+str(mcs)+".txt"
        self.pW.savePlotAsData(fileName)
    
class BarPlotSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)

    def start(self):        
        self.pW=CompuCellSetup.addNewPlotWindow(_title='Bar Plot',_xAxisTitle='Growth of US GDP',_yAxisTitle='Number of Suits')        
    def step(self,mcs):
    
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
    