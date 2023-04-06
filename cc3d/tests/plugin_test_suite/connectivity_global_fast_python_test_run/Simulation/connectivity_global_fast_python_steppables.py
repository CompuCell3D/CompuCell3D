from cc3d.core.PySteppables import *


class ConnectivitySteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        
    def start(self):
        # will turn connectivity for first 100 cells
        for cell in self.cell_list:
            if cell.id < 100:
                cell.connectivityOn = True                    

