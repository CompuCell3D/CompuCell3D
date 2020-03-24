from cc3d import CompuCellSetup

from diffusion_2D_uptakeSteppables import diffusion_2D_uptakeSteppable

CompuCellSetup.register_steppable(steppable=diffusion_2D_uptakeSteppable(frequency=1))

CompuCellSetup.run()
