
from cc3d import CompuCellSetup
        

from PermeableInterfaceDemoSteppables import PermeableInterfaceDemoSteppable

CompuCellSetup.register_steppable(steppable=PermeableInterfaceDemoSteppable(frequency=1))


CompuCellSetup.run()
