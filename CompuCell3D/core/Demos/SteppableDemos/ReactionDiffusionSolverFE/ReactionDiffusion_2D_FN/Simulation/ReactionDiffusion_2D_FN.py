from cc3d import CompuCellSetup

from ReactionDiffusion_2D_FNSteppables import ReactionDiffusion_2D_FNSteppable

CompuCellSetup.register_steppable(steppable=ReactionDiffusion_2D_FNSteppable(frequency=1))

CompuCellSetup.run()
