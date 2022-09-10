
from cc3d import CompuCellSetup
        

from HeterogeneousBCDemoSteppables import HeterogeneousBCDemoSteppable

CompuCellSetup.register_steppable(steppable=HeterogeneousBCDemoSteppable(frequency=1))


CompuCellSetup.run()
