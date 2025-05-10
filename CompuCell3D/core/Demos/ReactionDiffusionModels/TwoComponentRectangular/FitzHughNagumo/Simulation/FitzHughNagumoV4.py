
from cc3d import CompuCellSetup
        

from FitzHughNagumoV4Steppables import FitzHughNagumoV4Steppable

CompuCellSetup.register_steppable(steppable=FitzHughNagumoV4Steppable(frequency=1))


CompuCellSetup.run()
