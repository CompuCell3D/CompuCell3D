import CompuCellSetup
import os
import sys
import CompuCell
from PySteppables import *
from PlayerPython import *
from math import *

# import bionetAPI
class DeltaNotchClass(SteppableBasePy):
  def __init__(self,_simulator,_frequency):
    SteppableBasePy.__init__(self,_simulator,_frequency)
    # bionetAPI.initializeBionetworkManager(self.simulator)
    
  def start(self):
      
    modelFile='Simulation/DN_Collier.sbml'  
    self.addSBMLToCellTypes(_modelFile=modelFile,_modelName='DN',_types=[self.TYPEA],_stepSize=0.2)  
    
      
#     #Loading model
#     Name = 'DeltaNotch'
#     Key  = 'DN'
    
    
    
#     simulationDir=os.path.dirname (os.path.abspath( __file__ ))
#     Path= os.path.join(simulationDir,'DN_Collier.sbml')
#     Path=os.path.abspath(Path) # normalizing path
    
#     IntegrationStep = 0.2
#     bionetAPI.loadSBMLModel(Name, Path, Key, IntegrationStep)
    
#     bionetAPI.addSBMLModelToTemplateLibrary(Name,'TypeA')
#     bionetAPI.initializeBionetworks()
    
    #Initial conditions
    import random 
    
    state={} #dictionary to store state veriables of the SBML model
    
    for cell in self.cellList:
      if (cell):
        state['D'] = random.uniform(0.9,1.0)
        state['N'] = random.uniform(0.9,1.0)        
        self.setSBMLState(_modelName='DN',_cell=cell,_state=state)
#         bionetAPI.setBionetworkState(cell.id,'DeltaNotch',state) 
        
        cellDict=CompuCell.getPyAttrib(cell)
        cellDict['D']=state['D']
        cellDict['N']=state['N']

  def step(self,mcs):
    for cell in self.cellList:
      if (cell): 
        D=0.0; nn=0
        cellNeighborList=self.getCellNeighbors(cell)
        for neighbor in cellNeighborList:
          if (neighbor.neighborAddress):
            nn+=1
            state=self.getSBMLState(_modelName='DN',_cell=neighbor.neighborAddress)
#             state=bionetAPI.getBionetworkState(neighbor.neighborAddress.id,'DeltaNotch')
            D+=state['D']   
        if (nn>0):
          D=D/nn
          
        state={}  
        state['Davg']=D        
        self.setSBMLState(_modelName='DN',_cell=cell,_state=state)
        
#         bionetAPI.setBionetworkState(cell.id,'DeltaNotch',state) 
        
#         state=bionetAPI.getBionetworkState(cell.id,'DeltaNotch')
        state=self.getSBMLState(_modelName='DN',_cell=cell)
        cellDict=CompuCell.getPyAttrib(cell)
        cellDict['D']=D
        cellDict['N']=state['N']   
    self.timestepSBML()    
#     bionetAPI.timestepBionetworks() 


    
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
        cellDict=CompuCell.getPyAttrib(cell)
        self.scalarFieldDelta[cell]=cellDict['D']
        self.scalarFieldNotch[cell]=cellDict['N']
