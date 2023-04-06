
from cc3d import CompuCellSetup
        

from TypeDiffusivityAndStabilitySteppables import TypeDiffusivityAndStabilitySteppable

CompuCellSetup.register_steppable(steppable=TypeDiffusivityAndStabilitySteppable(frequency=1))


CompuCellSetup.run()
