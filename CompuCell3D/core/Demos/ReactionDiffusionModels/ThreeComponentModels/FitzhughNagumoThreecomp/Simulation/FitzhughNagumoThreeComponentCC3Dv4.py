
from cc3d import CompuCellSetup
        

from FitzhughNagumoThreeComponentCC3Dv4Steppables import FitzhughNagumoThreeComponentCC3Dv4Steppable

CompuCellSetup.register_steppable(steppable=FitzhughNagumoThreeComponentCC3Dv4Steppable(frequency=1))


CompuCellSetup.run()
