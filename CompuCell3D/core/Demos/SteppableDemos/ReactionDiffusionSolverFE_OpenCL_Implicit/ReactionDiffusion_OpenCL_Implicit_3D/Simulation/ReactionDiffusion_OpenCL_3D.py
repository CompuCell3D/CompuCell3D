
from cc3d import CompuCellSetup
        

from ReactionDiffusion_OpenCL_3DSteppables import ReactionDiffusion_OpenCL_3DSteppable

CompuCellSetup.register_steppable(steppable=ReactionDiffusion_OpenCL_3DSteppable(frequency=1))


CompuCellSetup.run()
