from cc3d import CompuCellSetup

from diffusion_2D_scaleSteppables import diffusion_2D_scaleSteppable

CompuCellSetup.register_steppable(steppable=diffusion_2D_scaleSteppable(frequency=1))

CompuCellSetup.run()
