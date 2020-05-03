
from cc3d import CompuCellSetup
        

from SimpleDiffusionSteppables import SimpleDiffusionSteppable

CompuCellSetup.register_steppable(steppable=SimpleDiffusionSteppable(frequency=1))


CompuCellSetup.run()
