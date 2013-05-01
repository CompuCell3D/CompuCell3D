
from PySteppables import *
import CompuCell
import sys
class BuildWall3DSteppable(SteppableBasePy):    

    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)
    def start(self):
        
        self.buildWall(self.WALL)

    def step(self,mcs):        
        print 'MCS=',mcs  
#         if mcs==2:      
#             self.destroyWall()
        
        if mcs==4:
            self.destroyWall()
            self.resizeAndShiftLattice(_newSize=(80,80,80), _shiftVec=(10,10,10))
            self.buildWall(self.WALL)
        
        
    def finish(self):
        # Finish Function gets called after the last MCS
        pass
        