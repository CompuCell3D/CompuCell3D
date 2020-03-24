from cc3d import CompuCellSetup

from diffusion_fast_boxSteppables import diffusion_fast_boxSteppable

CompuCellSetup.register_steppable(steppable=diffusion_fast_boxSteppable(frequency=1))

CompuCellSetup.run()
