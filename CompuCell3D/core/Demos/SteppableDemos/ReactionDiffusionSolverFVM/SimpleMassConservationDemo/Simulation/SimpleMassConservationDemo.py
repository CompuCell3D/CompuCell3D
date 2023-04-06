
from cc3d import CompuCellSetup
        

from SimpleMassConservationDemoSteppables import SimpleMassConservationDemoSteppable

CompuCellSetup.register_steppable(steppable=SimpleMassConservationDemoSteppable(frequency=1))


CompuCellSetup.run()
