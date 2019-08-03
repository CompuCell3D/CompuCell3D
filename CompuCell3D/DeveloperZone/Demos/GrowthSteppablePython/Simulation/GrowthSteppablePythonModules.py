from cc3d.core.PySteppables import *
from cc3d.cpp import CompuCellExtraModules

class GrowthSteppablePython(SteppableBasePy):

    def __init__(self,frequency=1):

        SteppableBasePy.__init__(self,frequency)
        self.growth_steppable_cpp = None

    def start(self):
        self.growth_steppable_cpp = CompuCellExtraModules.getGrowthSteppable()

    def step(self,mcs):
        
        if mcs == 10:
            
            self.growth_steppable_cpp.setGrowthRate(1,-1.2)



        