from PySteppables import *
import CompuCell
import sys

class ConnectivityElongationSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        
    def start(self):
        for cell in self.cellList:
            if cell.type==1:                
				cell.connectivityOn = True    
                
            elif cell.type==2:                
				cell.connectivityOn = True    
                
                
                

    def step(self,mcs):
        for cell in self.cellList:
            if cell.type==1:
                self.lengthConstraintPlugin.setLengthConstraintData(cell,20,20) # cell , lambdaLength, targetLength

                
            elif cell.type==2:
                self.lengthConstraintPlugin.setLengthConstraintData(cell,20,30)  # cell , lambdaLength, targetLength

                
                print "targetLength=",self.lengthConstraintLocalFlexPlugin.getTargetLength(cell)," lambdaLength=",self.lengthConstraintLocalFlexPlugin.getLambdaLength(cell)
                

