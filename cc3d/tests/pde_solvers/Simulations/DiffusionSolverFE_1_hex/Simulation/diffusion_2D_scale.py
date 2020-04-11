from cc3d import CompuCellSetup


        
from diffusion_steppable import DiffusionSolverSteppable
CompuCellSetup.register_steppable(steppable=DiffusionSolverSteppable(frequency=1))

CompuCellSetup.run()
