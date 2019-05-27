import cc3d.CompuCellSetup as CompuCellSetup
from .SBMLSolverSteppables import SBMLSolverSteppable


CompuCellSetup.register_steppable(steppable=SBMLSolverSteppable(frequency=1))


CompuCellSetup.run()

