
from PySteppables import *
import CompuCell
import sys
class ContactOrientationSteppable(SteppableBasePy):    

    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)
    def start(self):
        
        # iterating over all cells in simulation        
        for cell in self.cellList:
            self.contactOrientationPlugin.setOriantationVector(cell,0.,1.,0.)
            self.contactOrientationPlugin.setAlpha(cell,8.0)

        
    def step(self,mcs):        
        #type here the code that will run every _frequency MCS
        for cell in self.cellList:
            print "cell.id=",cell.id
            print 'alpha=',self.contactOrientationPlugin.getAlpha(cell)
    def finish(self):
        # Finish Function gets called after the last MCS
        pass
        