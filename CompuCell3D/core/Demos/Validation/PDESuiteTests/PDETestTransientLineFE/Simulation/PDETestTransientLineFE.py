
from cc3d import CompuCellSetup
        

from PDETestTransientLineFESteppables import PDETestTransientLineFESteppable

CompuCellSetup.register_steppable(steppable=PDETestTransientLineFESteppable(frequency=100))


CompuCellSetup.run()
