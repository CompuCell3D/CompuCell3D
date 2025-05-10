from cc3d import CompuCellSetup

from FitzHughNagumoSteppables import FitzHughNagumoSteppable

CompuCellSetup.register_steppable(steppable=FitzHughNagumoSteppable(frequency=1))

CompuCellSetup.run()
