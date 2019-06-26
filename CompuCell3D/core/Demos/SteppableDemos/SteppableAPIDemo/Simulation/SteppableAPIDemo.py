
from cc3d import CompuCellSetup
        

from SteppableAPIDemoSteppables import SteppableAPIDemoSteppable

CompuCellSetup.register_steppable(steppable=SteppableAPIDemoSteppable(frequency=1))


CompuCellSetup.run()
