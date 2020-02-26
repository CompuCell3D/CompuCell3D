
from cc3d import CompuCellSetup
        

from VariableDiffDemoSteppables import VariableDiffDemoSteppable

CompuCellSetup.register_steppable(steppable=VariableDiffDemoSteppable(frequency=1))


CompuCellSetup.run()
