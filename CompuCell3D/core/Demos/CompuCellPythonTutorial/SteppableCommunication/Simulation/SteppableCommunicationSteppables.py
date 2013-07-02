
from PySteppables import *
import CompuCell
import sys

class SteppableCommunicationSteppable(SteppableBasePy):    

    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        
    def step(self,mcs):        
        extraSteppable=self.getSteppableByClassName('ExtraSteppable')
        print 'extraSteppable.sharedParameter=',extraSteppable.sharedParameter
        

class ExtraSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)        
        self.sharedParameter=25
        
    def step(self,mcs):
        print "ExtraSteppable: This function is called every 1 MCS"
            
    
