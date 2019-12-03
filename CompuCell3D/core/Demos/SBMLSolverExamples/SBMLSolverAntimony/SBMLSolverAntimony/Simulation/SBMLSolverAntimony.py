from cc3d import CompuCellSetup
from SBMLSolverAntimonySteppables import SBMLSolverSteppable
from SBMLSolverAntimonySteppables import IdFieldVisualizationSteppable


CompuCellSetup.register_steppable(steppable=SBMLSolverSteppable(frequency=1))
CompuCellSetup.register_steppable(steppable=IdFieldVisualizationSteppable(frequency=1))


CompuCellSetup.run()
