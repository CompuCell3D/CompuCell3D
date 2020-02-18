from cc3d import CompuCellSetup

from diffusion_2D_ADESteppables import diffusion_2D_ADESteppable

CompuCellSetup.register_steppable(steppable=diffusion_2D_ADESteppable(frequency=1))

CompuCellSetup.run()
