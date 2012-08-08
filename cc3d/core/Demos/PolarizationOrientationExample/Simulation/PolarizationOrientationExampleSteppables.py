from PySteppables import *
import CompuCell
import sys
from XMLUtils import dictionaryToMapStrStr as d2mss



            
class PolarizationOrientationExampleSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=100):
        SteppableBasePy.__init__(self,_simulator,_frequency)
        

    def start(self):
        for cell in self.cellList:
            self.polarizationVectorPlugin.setPolarizationVector(cell,1,1,0);
            # uncomment this line if you wan to use cell id - based lambdaConcentration. Make sure you do not list any tags inside <Plugin Name="CellOrientation"> element in the xml file
            # self.cellOrientationPlugin.setLambdaCellOrientation(cell,50.0) 
            
    def step(self,mcs):pass
                
        

