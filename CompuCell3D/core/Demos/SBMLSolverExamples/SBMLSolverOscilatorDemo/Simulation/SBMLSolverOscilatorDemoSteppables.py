from PySteppables import *
import CompuCell
import sys
import os

class SBMLSolverOscilatorDemoSteppable(SteppableBasePy):    

    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        self.pW = None
        
    def start(self):
        self.pW=self.addNewPlotWindow(_title='S1 concentration',_xAxisTitle='MonteCarlo Step (MCS)',_yAxisTitle='Variables')
        self.pW.addPlot('S1',_style='Dots',_color='red',_size=5)
        
        # iterating over all cells in simulation        
        for cell in self.cellList:
            # you can access/manipulate cell properties here
            cell.targetVolume=25
            cell.lambdaVolume=2.0
            
        #SBML SOLVER          
        
        # adding options that setup SBML solver integrator - these are optional but useful when encounteting integration instabilities       
        options={'relative':1e-10,'absolute':1e-12}
        # options={'relative':1e-10,'absolute':1e-12}
        self.setSBMLGlobalOptions(options)
        
        modelFile='Simulation/oscli.sbml' # this can be e.g. partial path 'Simulation/osci.sbml'
        stepSize=0.02
            
        initialConditions={}
        initialConditions['S1']=0.0
        initialConditions['S2']=1.0
        self.addSBMLToCellTypes(_modelFile=modelFile,_modelName='OSCIL',_types=[self.NONCONDENSING],_stepSize=stepSize,_initialConditions=initialConditions) 
        
    def step(self,mcs):        
        if not self.pW:
            self.pW=self.addNewPlotWindow(_title='S1 concentration',_xAxisTitle='MonteCarlo Step (MCS)',_yAxisTitle='Variables')
            self.pW.addPlot('S1',_style='Dots',_color='red',_size=5)
        
        added=False
        for cell in self.cellList:
            if cell.type==self.NONCONDENSING:     
                state=self.getSBMLState(_modelName='OSCIL',_cell=cell) 
                concentration=state['S1']
                cell.targetVolume=25+10*concentration
                
                if not added:
                    self.pW.addDataPoint("S1",mcs,concentration) 
                    added=True
                    
        self.pW.showAllPlots()         
        self.timestepSBML()
        
