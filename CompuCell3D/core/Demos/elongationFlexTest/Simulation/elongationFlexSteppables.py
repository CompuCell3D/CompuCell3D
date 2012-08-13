from PySteppables import *
import CompuCell
import sys

class ElongationFlexSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self, _simulator, _frequency)
        # self.lengthConstraintPlugin=CompuCell.getLengthConstraintPlugin()
        
        
    def start(self):
        pass

    def step(self,mcs):
        for cell in self.cellList:
            if cell.type==1:
                self.lengthConstraintPlugin.setLengthConstraintData(cell,20,20) # cell , lambdaLength, targetLength
                self.connectivityGlobalPlugin.setConnectivityStrength(cell,10000000) #cell, strength
                
            elif cell.type==2:
                self.lengthConstraintPlugin.setLengthConstraintData(cell,20,30)  # cell , lambdaLength, targetLength
                self.connectivityGlobalPlugin.setConnectivityStrength(cell,10000000) #cell, strength
                
                # print "targetLength=",self.lengthConstraintFlexPlugin.getTargetLength(cell)," lambdaLength=",self.lengthConstraintFlexPlugin.getLambdaLength(cell)
                

