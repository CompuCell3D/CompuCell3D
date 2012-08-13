import os
import sys
import CompuCell
from PySteppables import *
from PlayerPython import *
from math import *

import bionetAPI
class DeltaNotchClass(SteppableBasePy):
  def __init__(self,_simulator,_frequency):
    SteppableBasePy.__init__(self,_simulator,_frequency)
    bionetAPI.initializeBionetworkManager(self.simulator)
    
  def start(self):
    #Loading model
    Name = "DeltaNotch"
    Key  = "DN"
    # Path = os.getcwd()+"\DeltaNotch\DN_Collier.sbml"
    simulationDir=os.path.dirname (os.path.abspath( __file__ ))
    Path= os.path.join(simulationDir,"DN_Collier.sbml")
    Path=os.path.abspath(Path) # normalizing path
    
    IntegrationStep = 0.2
    bionetAPI.loadSBMLModel(Name, Path, Key, IntegrationStep)
    
    bionetAPI.addSBMLModelToTemplateLibrary(Name,"TypeA")
    bionetAPI.initializeBionetworks()
    
    #Initial conditions
    import random 
    for cell in self.cellList:
      if (cell):
        D = random.uniform(0.9,1.0)
        N = random.uniform(0.9,1.0)
        bionetAPI.setBionetworkValue("DN_D",D,cell.id)
        bionetAPI.setBionetworkValue("DN_N",N,cell.id)
        cellDict=CompuCell.getPyAttrib(cell)
        cellDict["D"]=D
        cellDict["N"]=N

  def step(self,mcs):
    for cell in self.cellList:
      if (cell): 
        D=0.0; nn=0
        cellNeighborList=self.getCellNeighbors(cell)
        for neighbor in cellNeighborList:
          if (neighbor.neighborAddress):
            nn+=1
            D+=bionetAPI.getBionetworkValue("DN_D",neighbor.neighborAddress.id)
        if (nn>0):
          D=D/nn
        bionetAPI.setBionetworkValue("DN_Davg",D,cell.id)
        cellDict=CompuCell.getPyAttrib(cell)
        cellDict["D"]=D
        cellDict["N"]=bionetAPI.getBionetworkValue("DN_N",cell.id) 
    bionetAPI.timestepBionetworks() 

    
class ExtraFields(SteppableBasePy):
  def __init__(self,_simulator,_frequency=1):
    SteppableBasePy.__init__(self,_simulator,_frequency)
   
  def setScalarFields(self,_field1,_field2):
    self.scalarField1=_field1
    self.scalarField2=_field2  

  def step(self,mcs):
    clearScalarValueCellLevel(self.scalarField1)
    clearScalarValueCellLevel(self.scalarField2)
    for cell in self.cellList:
      if (cell):
        cellDict=CompuCell.getPyAttrib(cell)
        fillScalarValueCellLevel(self.scalarField1,cell,cellDict["D"])
        fillScalarValueCellLevel(self.scalarField2,cell,cellDict["N"])