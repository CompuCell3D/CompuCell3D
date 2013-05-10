from PySteppables import *
import CompuCell
import sys
import os

class SBMLSolverSteppable(SteppableBasePy):    
    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        
    def start(self):
        
        modelFile='Simulation/test_1.xml'
        
        initialConditions={}
        initialConditions['S1']=0.00020
        initialConditions['S2']=0.000002

        self.addSBMLToCellIds(_modelFile=modelFile,_modelName='dupa',_ids=range(1,11),_stepSize=0.5,_initialConditions=initialConditions)
        
        self.addFreeFloatingSBML(_modelFile=modelFile,_modelName='Medium_dupa',_stepSize=0.5,_initialConditions=initialConditions)
        self.addFreeFloatingSBML(_modelFile=modelFile,_modelName='Medium_dupa1',_stepSize=0.5,_initialConditions=initialConditions)
        self.addFreeFloatingSBML(_modelFile=modelFile,_modelName='Medium_dupa2')
        self.addFreeFloatingSBML(_modelFile=modelFile,_modelName='Medium_dupa3')
        self.addFreeFloatingSBML(_modelFile=modelFile,_modelName='Medium_dupa4')
        
        cell20=self.inventory.attemptFetchingCellById(20)        
        
        self.addSBMLToCell(_modelFile=modelFile,_modelName='dupa',_cell=cell20)
                
    def step(self,mcs):        
        
        self.timestepSBML()
        
        cell10=self.inventory.attemptFetchingCellById(10)
        print 'cell=',cell10
        
        speciesDict=self.getSBMLState(_modelName='Medium_dupa2')
        print 'speciesDict=',speciesDict.values()

        state={}
        state['S1']=10
        state['S2']=0.5
        if mcs==3:
            self.setSBMLState('Medium_dupa2',_state=state)
        
#         if mcs==5:
#             self.deleteSBMLFromCellIds(_modelName='dupa',_ids=range(1,11))
            

        if mcs==7:
            cell25=self.inventory.attemptFetchingCellById(25)
            self.copySBMLs(_fromCell=cell10,_toCell=cell25)

