import cc3d.CompuCellSetup as CompuCellSetup
from .SBMLSolverLegacySteppables import SBMLSolverSteppable


CompuCellSetup.register_steppable(steppable=SBMLSolverSteppable(frequency=1))


CompuCellSetup.run()

