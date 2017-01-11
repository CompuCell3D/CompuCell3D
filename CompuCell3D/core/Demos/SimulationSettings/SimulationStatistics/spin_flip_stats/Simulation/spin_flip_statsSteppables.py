
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

        accepted_pixel_copy_mask = self.get_accepted_pixel_copy_mask()
        attempted_pixel_copy_prob_array = self.get_attempted_pixel_copy_prob_array()
        attempted_pixel_copy_points = self.get_attempted_pixel_copy_points()
        
        print 'indicator whether pixel copy attempt was succesful = ',
        print accepted_pixel_copy_mask[:20]        
        
        print 'acceptance probabilities of flip attempts= ',
        print attempted_pixel_copy_prob_array[:20]
        
        print 'points at which pixel copy attempts took place flip points = ',
        print attempted_pixel_copy_points[:20]
        
        print 'Example of how to select accepted pixel copy points'
        accepted_pixel_copy_points = attempted_pixel_copy_points[accepted_pixel_copy_mask]
        print 'accepted_pixel_copy_points=',accepted_pixel_copy_points[:10]
        
        print 'Example how to print rejected  probabilities '
        rej_probs = attempted_pixel_copy_prob_array[~accepted_pixel_copy_mask] 
        print 'rejected probabilities = ',rej_probs[:10]
        

        print 'Example how to print accepted probabilities '
        acc_probs = attempted_pixel_copy_prob_array[accepted_pixel_copy_mask] 
        print 'accepted probabilities = ',acc_probs[:10]
        
        
        
    def finish(self):
        # Finish Function gets called after the last MCS
        pass
        