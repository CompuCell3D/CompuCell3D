from cc3d import CompuCellSetup

from ReactionDiffusion_2DSteppables import ReactionDiffusion_2DSteppable

CompuCellSetup.register_steppable(steppable=ReactionDiffusion_2DSteppable(frequency=1))

CompuCellSetup.run()
