from PySteppables import *
import CompuCell
import sys
from XMLUtils import dictionaryToMapStrStr as d2mss
            
class ChemotaxisSteering(SteppableBasePy):
    def __init__(self,_simulator,_frequency=100):
        SteppableBasePy.__init__(self,_simulator,_frequency)
                
    def start(self):
        
        for cell in self.cellList:
            if cell.type==self.MACROPHAGE:
                cd=self.chemotaxisPlugin.addChemotaxisData(cell,"ATTR")
                cd.setLambda(30.0)                
                cd.assignChemotactTowardsVectorTypes([self.MEDIUM,self.BACTERIUM])
                break
    
    def step(self,mcs):        
        if mcs>100 and not mcs%100:
            for cell in self.cellList:
                if cell.type==self.MACROPHAGE:
        
                    cd=self.chemotaxisPlugin.getChemotaxisData(cell,"ATTR")
                    if cd:
                        lm=cd.getLambda()-3
                        cd.setLambda(lm)
                    break

                    
                    
                    
                    
