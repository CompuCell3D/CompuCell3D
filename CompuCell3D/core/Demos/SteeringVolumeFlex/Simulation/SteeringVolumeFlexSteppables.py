
from PySteppables import *
import CompuCell
import sys
class SteeringVolumeFlexSteppable(SteppableBasePy):    

    def __init__(self,_simulator,_frequency=1):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        
    def start(self):
        pass
        
    def step(self,mcs):        
        if mcs == 10:
            # read current attribute - here we are reading TargetVolume for cell type Condensing
            current_target_volume = float(
                self.getXMLAttributeValue(
                'TargetVolume',                 # attribute name
                ['Plugin','Name','Volume'],     # access path 
                ['VolumeEnergyParameters', 'CellType','Condensing']
                )
            )
            
            self.setXMLAttributeValue(
                'TargetVolume',                 # attribute name
                current_target_volume*2,        # new value of the attribute 
                ['Plugin','Name','Volume'],     # acces path
                ['VolumeEnergyParameters', 'CellType','Condensing']
            )            

            self.updateXML()    

    def finish(self):
        # Finish Function gets called after the last MCS
        pass
        