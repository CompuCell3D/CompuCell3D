
from cc3d import CompuCellSetup
        

from FluctuationCompensatorDemoSteppables import FluctuationCompensatorDemoSteppable

CompuCellSetup.register_steppable(steppable=FluctuationCompensatorDemoSteppable(frequency=100))


CompuCellSetup.run()
