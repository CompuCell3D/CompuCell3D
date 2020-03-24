from cc3d import CompuCellSetup

from diffusion_2D_kernelSteppables import diffusion_2D_kernelSteppable

CompuCellSetup.register_steppable(steppable=diffusion_2D_kernelSteppable(frequency=1))

CompuCellSetup.run()
