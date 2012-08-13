from PySteppables import *
import CompuCell
import sys

class ConnectivityElongationSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        
    def start(self):
        for cell in self.cellList:
            if cell.type==1:                
                self.connectivityGlobalPlugin.setConnectivityStrength(cell,20000000) #cell, strength
                
            elif cell.type==2:                
                self.connectivityGlobalPlugin.setConnectivityStrength(cell,10000000) #cell, strength
                
                

    def step(self,mcs):
        for cell in self.cellList:
            if cell.type==1:
                self.lengthConstraintLocalFlexPlugin.setLengthConstraintData(cell,20,20) # cell , lambdaLength, targetLength
                # self.connectivityGlobalPlugin.setConnectivityStrength(cell,20000000) #cell, strength
                
            elif cell.type==2:
                self.lengthConstraintLocalFlexPlugin.setLengthConstraintData(cell,20,30)  # cell , lambdaLength, targetLength
                # self.connectivityGlobalPlugin.setConnectivityStrength(cell,20000000) #cell, strength
                
                print "targetLength=",self.lengthConstraintLocalFlexPlugin.getTargetLength(cell)," lambdaLength=",self.lengthConstraintLocalFlexPlugin.getLambdaLength(cell)
                

