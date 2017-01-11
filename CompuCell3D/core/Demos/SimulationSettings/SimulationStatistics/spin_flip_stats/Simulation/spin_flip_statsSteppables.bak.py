
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
        #type here the code that will run every _frequency MCS
        
        en_calculator = self.potts.getEnergyFunctionCalculator()
#         print 'en_calculator=',dir(en_calculator)
        num_en_calcs = int(en_calculator.get_number_energy_fcn_calculations())
        
        print 'number of attampted en calculations=',num_en_calcs 
        
        a = en_calculator.range(10)
         
        print 'a=',type(a)
        
        probs = en_calculator.get_current_mcs_prob_npy_array(num_en_calcs)
        print 'type probs =', type(probs), probs.dtype
        print 'probs = ', probs[:20]

        acc = en_calculator.get_current_mcs_accepted_mask_npy_array(num_en_calcs)
        print 'type acc =', type(acc), acc.dtype
        print 'acc = ', acc[:20]

        flip_points = en_calculator.get_current_mcs_flip_attempt_points_npy_array(3*num_en_calcs)
        print 'type flip_points=', type(flip_points), flip_points.dtype
        print 'flip_points = ', flip_points[:60]
        print np.reshape(flip_points,(-1,3))[:5]
        
        print 'flip points = ',self.get_flip_points()[:5]
        print self.get_accepted_flips_mask()[:20]
        
        print self.get_flip_prob_array()[:20]
        

        
        
#         accepted_mask = np.zeros((num_en_calcs,), dtype=np.bool)
#         en_calculator.request_current_mcs_accepted_mask_array(accepted_mask)
        
#         print 'accepted_mask=', accepted_mask[:20]
        
        
        
        
#         for cell in self.cellList:
#             print "cell.id=",cell.id
    def finish(self):
        # Finish Function gets called after the last MCS
        pass
        