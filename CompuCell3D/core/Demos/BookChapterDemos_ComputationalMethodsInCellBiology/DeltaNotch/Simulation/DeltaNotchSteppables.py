import CompuCellSetup
import os
import sys
import CompuCell
from PySteppables import *
from PlayerPython import *
from math import *
import random 

# import bionetAPI
class DeltaNotchClass(SteppableBasePy):
    def __init__(self,_simulator,_frequency):
        SteppableBasePy.__init__(self,_simulator,_frequency)
    def start(self):
        
        # adding options that setup SBML solver integrator - these are optional but useful when encounteting integration instabilities              
        options={'relative':1e-10,'absolute':1e-12}
        self.setSBMLGlobalOptions(options)
        
        modelFile='Simulation/DN_Collier.sbml'  
        self.addSBMLToCellTypes(_modelFile=modelFile,_modelName='DN',_types=[self.TYPEA],_stepSize=0.2)  
        
        state={} #dictionary to store state veriables of the SBML model
    
        for cell in self.cellList:
      
            state['D'] = random.uniform(0.9,1.0)
            state['N'] = random.uniform(0.9,1.0)        
            self.setSBMLState(_modelName='DN',_cell=cell,_state=state)
                        
            cell.dict['D']=state['D']
            cell.dict['N']=state['N']
        
    def step(self,mcs):
        for cell in self.cellList:

            D=0.0; nn=0        
            for neighbor , commonSurfaceArea in self.getCellNeighborDataList(cell):
                if neighbor:
                    nn+=1
                    state=self.getSBMLState(_modelName='DN',_cell=neighbor)
                
                    D+=state['D']   
            if (nn>0):
                D=D/nn
              
            state={}  
            state['Davg']=D        
            self.setSBMLState(_modelName='DN',_cell=cell,_state=state)
            
            state=self.getSBMLState(_modelName='DN',_cell=cell)    
            cell.dict['D']=D
            cell.dict['N']=state['N']   
        self.timestepSBML()    



    
class ExtraFields(SteppableBasePy):
  def __init__(self,_simulator,_frequency=1):
    SteppableBasePy.__init__(self,_simulator,_frequency)
    
    self.scalarFieldDelta=CompuCellSetup.createScalarFieldCellLevelPy("Delta")
    self.scalarFieldNotch=CompuCellSetup.createScalarFieldCellLevelPy("Notch")
   

  def step(self,mcs):
    self.scalarFieldDelta.clear()
    self.scalarFieldNotch.clear()
    
    for cell in self.cellList:
      if cell:        
        self.scalarFieldDelta[cell]=cell.dict['D']
        self.scalarFieldNotch[cell]=cell.dict['N']
