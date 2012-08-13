
from PySteppables import *
import CompuCell
import sys
import os
import bionetAPI

class BionetDemoSteppable(SteppableBasePy):    

    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        bionetAPI.initializeBionetworkManager(self.simulator)
    def start(self):
        
        # iterating over all cells in simulation        
        for cell in self.cellList:
            # you can access/manipulate cell properties here
            cell.targetVolume=25
            cell.lambdaVolume=2.0

        #bionet section        
        modelName = "OSCLI"
        modelNickname  = "OSC" # this is usually shorter version version of model name
        
        fileDir=os.path.dirname (os.path.abspath( __file__ ))
        
        modelPath=os.path.join(fileDir,"oscli.sbml") 
        print "Path=",modelPath        
        
        integrationStep = 0.02
        bionetAPI.loadSBMLModel(modelName , modelPath, modelNickname, integrationStep)
        
        bionetAPI.addSBMLModelToTemplateLibrary("OSCLI","NonCondensing")
        
        bionetAPI.initializeBionetworks()
        
        
        # iterating over all cells in simulation        
        for cell in self.cellList:         
            if cell.type==self.NONCONDENSING:
                bionetAPI.setBionetworkValue("OSC_S1",0,cell.id)
                bionetAPI.setBionetworkValue("OSC_S2",1,cell.id)
            
        
        
        
        
    def step(self,mcs):        
        pass
        #type here the code that will run every _frequency MCS
        for cell in self.cellList:
            if cell.type==self.NONCONDENSING:            
                concentration=bionetAPI.getBionetworkValue("S1",cell.id)
#                 print "concentration=",concentration
                cell.targetVolume=25+10*concentration
        
        bionetAPI.timestepBionetworks() 
                
            
    def finish(self):
        # Finish Function gets called after the last MCS
        pass
        
from PySteppables import *
import CompuCell
import sys

from PlayerPython import *
import CompuCellSetup
from math import *


class PlotsSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        self.pW=None
        
    def initPlot(self):
        print "PlotsSteppable: This function is called once before simulation"
        
        import CompuCellSetup  
        self.pW=CompuCellSetup.viewManager.plotManager.getNewPlotWindow()
        if not self.pW:
            return
        #Plot Title - properties           
        self.pW.setTitle("OSCLI")
        self.pW.setTitleSize(12)
        self.pW.setTitleColor("Green") # you may choose different color - type its name
        
        #plot background
        self.pW.setPlotBackgroundColor("orange") # you may choose different color - type its name
        
        # properties of x axis
        self.pW.setXAxisTitle("TITLE OF Y AXIS")
        self.pW.setXAxisTitleSize(10)      
        self.pW.setXAxisTitleColor("blue")  # you may choose different color - type its name            
        
        # properties of y axis
        self.pW.setYAxisTitle("TITLE OF Y AXIS")        
        self.pW.setYAxisLogScale()
        self.pW.setYAxisTitleSize(10)        
        self.pW.setYAxisTitleColor("red")  # you may choose different color - type its name                                
        
        # choices for style are NoCurve,Lines,Sticks,Steps,Dots
        self.pW.addPlot("DATA_SERIES_1",_style='Dots')
        #self.pW.addPlot("DATA SERIES 2",_style='Steps') # you may add more than one data series
        
        # plot MCS
        self.pW.changePlotProperty("DATA_SERIES_1","LineWidth",5)
        self.pW.changePlotProperty("DATA_SERIES_1","LineColor","red")     
        
        self.pW.addGrid()
        #adding automatically generated legend
        # default possition is at the bottom of the plot but here we put it at the top
        self.pW.addAutoLegend("top")
        
        self.clearFlag=False
        
    def start(self):
        
        self.initPlot()        
        
    def step(self,mcs):
        if not self.pW:
            self.initPlot()        
            
        print "PlotsSteppable: This function is called every 10 MCS"
        for cell in self.cellList:
            
            if cell.type==self.NONCONDENSING:
                
                concentration=bionetAPI.getBionetworkValue("OSC_S1",cell.id)
                
                #self.pW.eraseAllData() # this is how you erase previous content of the plot
                self.pW.addDataPoint("DATA_SERIES_1",mcs,concentration) # arguments are (name of the data series, x, y)
                print "concentration=",concentration
                break
                
        self.pW.showAllPlots()        
            
    def finish(self):
        # this function may be called at the end of simulation - used very infrequently though
        return
    
