from PySteppables import *
import CompuCell
import sys
import os

class SBMLSolverSteppable(SteppableBasePy):    
    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        
    def start(self):

        # adding options that setup SBML solver integrator - these are optional but useful when encounteting integration instabilities              
        options={'relative':1e-10,'absolute':1e-12}
        self.setSBMLGlobalOptions(options)

       
        modelFile='Simulation/test_1.xml'
        
        initialConditions={}
        initialConditions['S1']=0.00020
        initialConditions['S2']=0.000002

        self.addSBMLToCellIds(_modelFile=modelFile,_modelName='dp',_ids=range(1,11),_stepSize=0.5,_initialConditions=initialConditions)
        
        self.addFreeFloatingSBML(_modelFile=modelFile,_modelName='Medium_dp',_stepSize=0.5,_initialConditions=initialConditions)
        self.addFreeFloatingSBML(_modelFile=modelFile,_modelName='Medium_dp1',_stepSize=0.5,_initialConditions=initialConditions)
        self.addFreeFloatingSBML(_modelFile=modelFile,_modelName='Medium_dp2')
        self.addFreeFloatingSBML(_modelFile=modelFile,_modelName='Medium_dp3')
        self.addFreeFloatingSBML(_modelFile=modelFile,_modelName='Medium_dp4')
        
        cell20=self.attemptFetchingCellById(20)        
        
        self.addSBMLToCell(_modelFile=modelFile,_modelName='dp',_cell=cell20)
        
                
    def step(self,mcs):        
        
        self.timestepSBML()
        
        cell10=self.inventory.attemptFetchingCellById(10)
        print 'cell=',cell10
        
        state=self.getSBMLState(_modelName='Medium_dp2')
        print 'state=',state.values()

        state={}
        state['S1']=10
        state['S2']=0.5
        if mcs==3:
            self.setSBMLState('Medium_dp2',_state=state)
        
#         if mcs==5:
#             self.deleteSBMLFromCellIds(_modelName='dp',_ids=range(1,11))
            

        if mcs==7:
            cell25=self.inventory.attemptFetchingCellById(25)
            self.copySBMLs(_fromCell=cell10,_toCell=cell25)

