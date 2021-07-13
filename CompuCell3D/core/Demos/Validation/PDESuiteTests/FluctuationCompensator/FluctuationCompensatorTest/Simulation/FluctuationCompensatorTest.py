
from cc3d import CompuCellSetup
        

from FluctuationCompensatorTestSteppables import FluctuationCompensatorTestSteppable

CompuCellSetup.register_steppable(steppable=FluctuationCompensatorTestSteppable(frequency=100))


CompuCellSetup.run()
