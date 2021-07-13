
from cc3d import CompuCellSetup
        

from diffusion_testSteppables import DiffusionTestSteppable

CompuCellSetup.register_steppable(steppable=DiffusionTestSteppable(frequency=1))


CompuCellSetup.run()
