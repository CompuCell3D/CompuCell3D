
from PySteppables import *
import CompuCell
import sys
import numpy as np

class spin_flip_statsSteppable(SteppableBasePy):    

    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)
    def start(self):
        # any code in the start function runs before MCS=0
        pass
    def step(self,mcs):        

        print 'indicator whether pixel copy attempt was succesful = ',
        print self.get_accepted_pixel_copy_mask()[:20]        
        
        print 'acceptance probabilities of flip attempts= ',
        print self.get_attempted_pixel_copy_prob_array()[:20]
        
        print 'points at which pixel copy attempts took place flip points = ',
        print self.get_attempted_pixel_copy_points()[:20]
        
    def finish(self):
        # Finish Function gets called after the last MCS
        pass
        