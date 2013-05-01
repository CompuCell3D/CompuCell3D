
from PySteppables import *
import CompuCell
import sys
class DynamicNumberOfProcessorsSteppable(SteppableBasePy):    

    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        
    def step(self,mcs):  
        if mcs==10:
            self.resizeAndShiftLattice(_newSize=(400,400,1), _shiftVec=(100,100,0))
#             self.changeNumberOfWorkNodes(4)    

        if mcs==100:    
            self.changeNumberOfWorkNodes(8)    
        