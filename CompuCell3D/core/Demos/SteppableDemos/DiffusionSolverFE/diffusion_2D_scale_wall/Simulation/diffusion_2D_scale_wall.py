from cc3d import CompuCellSetup

from diffusion_2D_scale_wallSteppables import diffusion_2D_scale_wallSteppable

CompuCellSetup.register_steppable(steppable=diffusion_2D_scale_wallSteppable(frequency=1))

CompuCellSetup.run()
