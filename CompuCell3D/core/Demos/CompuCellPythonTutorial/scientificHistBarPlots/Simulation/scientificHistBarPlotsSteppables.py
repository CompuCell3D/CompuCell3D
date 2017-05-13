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
        self.pW=self.addNewPlotWindow(_title='Histogram of Cell Volumes',_xAxisTitle='Number of Cells',_yAxisTitle='Volume Size in Pixels')
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
        
        self.pW.addHistogram(plot_name='Hist 1' , value_array = gauss ,  number_of_bins=10)        
        self.pW.addHistogram(plot_name='Hist 2' , value_array = volList ,  number_of_bins=10)
        self.pW.addHistogram(plot_name='Hist 3' , value_array = volList ,  number_of_bins=50)                

        fileName="HistPlots_"+str(mcs)+".txt"
        self.pW.savePlotAsData(fileName,CSV_FORMAT)

        fileName="HistPlots_"+str(mcs)+".png"
        self.pW.savePlotAsPNG(fileName,1000,1000) # here we specify size of the image saved - default is 400 x 400
    
class BarPlotSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)

    def start(self):        
    
        self.pW = self.addNewPlotWindow(_title='Bar Plot',_xAxisTitle='Growth of US GDP',_yAxisTitle='Number of Suits')        
        self.pW.addPlot(_plotName='GDP',_color='red',_style='bars', _size=0.5)
        
    def step(self,mcs):
    
        if (mcs%20 == 0):
        
            self.pW.eraseAllData()
            
            gdpList = []
            locations = []
            for i in range(6):
                gdpList.append(random.uniform(1, 100))
                locations.append(random.uniform(1, 20))
            
            self.pW.addDataSeries('GDP',locations,gdpList)    
            
            
            # for gdp, loc in zip(gdpList,locations):        
            
                # self.pW.addDataPoint('GDP',loc,gdp)
                
            # self.pW.addBarPlotData(gdpList,locations,1)
        
    
        fileName="BarPlots_"+str(mcs)+".png"
        self.pW.savePlotAsPNG(fileName,1000,1000) # here we specify size of the image saved - default is 400 x 400

        fileName="BarPlots_"+str(mcs)+".txt"
        self.pW.savePlotAsData(fileName,CSV_FORMAT)
