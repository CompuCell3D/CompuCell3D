from cc3d import CompuCellSetup

from DiffusionSolverFE_3DSteppables import DiffusionSolverFE_3DSteppable

CompuCellSetup.register_steppable(steppable=DiffusionSolverFE_3DSteppable(frequency=1))

CompuCellSetup.run()
