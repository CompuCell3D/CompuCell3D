from cc3d import CompuCellSetup

from diffusion_2D_initSteppables import diffusion_2D_initSteppable

CompuCellSetup.register_steppable(steppable=diffusion_2D_initSteppable(frequency=1))

CompuCellSetup.run()
