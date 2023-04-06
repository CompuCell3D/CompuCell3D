
from cc3d import CompuCellSetup
        

from DiffusivityFieldDemoSteppables import DiffusivityFieldDemoSteppable

CompuCellSetup.register_steppable(steppable=DiffusivityFieldDemoSteppable(frequency=1))


CompuCellSetup.run()
