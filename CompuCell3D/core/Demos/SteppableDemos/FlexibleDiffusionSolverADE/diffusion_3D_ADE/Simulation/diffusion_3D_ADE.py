from cc3d import CompuCellSetup

from diffusion_3D_ADESteppables import diffusion_3D_ADESteppable

CompuCellSetup.register_steppable(steppable=diffusion_3D_ADESteppable(frequency=1))

CompuCellSetup.run()
