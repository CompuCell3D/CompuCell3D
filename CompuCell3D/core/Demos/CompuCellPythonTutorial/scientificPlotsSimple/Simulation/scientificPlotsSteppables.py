from PySteppables import *
import CompuCell
import CompuCellSetup
import time
import sys
            
class ExtraPlotSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)

    def start(self):
    
        self.pW=CompuCellSetup.addNewPlotWindow(_title='Average Volume And Surface',_xAxisTitle='MonteCarlo Step (MCS)',_yAxisTitle='Variables')
        self.pW.addPlot('MVol',_style='Dots',_color='red',_size=5)
        self.pW.addPlot('MSur',_style='Steps',_size=1)
        self.pW.setYAxisLogScale()        
        
    def step(self,mcs):
                
        meanSurface=0.0
        meanVolume=0.0
        numberOfCells=0
        for cell  in  self.cellList:
            meanVolume+=cell.volume
            meanSurface+=cell.surface
            numberOfCells+=1
        meanVolume/=float(numberOfCells)
        meanSurface/=float(numberOfCells)
                
        if  mcs >100 and mcs <200:
            self.pW.eraseAllData()
        else:
            self.pW.addDataPoint('MVol',mcs,meanVolume)
            self.pW.addDataPoint('MSur',mcs,meanSurface)
            if mcs>=200:
                print 'Adding meanVolume=',meanVolume
                print 'plotData=',self.pW.plotData['MVol']
            
        self.pW.showAllPlots()
                
        #Saving plots as PNG's
        if mcs<50:            
            fileName='ExtraPlots_'+str(mcs)+'.png'
            self.pW.savePlotAsPNG(fileName,1000,1000) # here we specify size of the image saved - default is 400 x 400
            self.pW.savePlotAsData(fileName+'.txt')
                
class ExtraMultiPlotSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)

    def start(self):
        
        self.pWVol=CompuCellSetup.addNewPlotWindow(_title='Average Volume',_xAxisTitle='MonteCarlo Step (MCS)',_yAxisTitle='Average Volume')        
        self.pWVol.addPlot(_plotName='MVol',_style='Dots',_color='red',_size=5)        
        self.pWSur=CompuCellSetup.addNewPlotWindow(_title='Average Surface',_xAxisTitle='MonteCarlo Step (MCS)',_yAxisTitle='Average Surface')                
        self.pWSur.addPlot(_plotName='MSur')
        
    def step(self,mcs):
        
        meanSurface=0.0
        meanVolume=0.0
        numberOfCells=0
        for cell  in  self.cellList:
            meanVolume+=cell.volume
            meanSurface+=cell.surface
            numberOfCells+=1
        meanVolume/=float(numberOfCells)
        meanSurface/=float(numberOfCells)
        
        self.pWVol.addDataPoint("MVol",mcs,meanVolume)
        self.pWSur.addDataPoint("MSur",mcs,meanSurface)
        print "meanVolume=",meanVolume,"meanSurface=",meanSurface
                
        self.pWVol.showAllPlots()
        self.pWSur.showAllPlots()
