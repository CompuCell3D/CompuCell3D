import cc3d.CompuCellSetup as CompuCellSetup
from .SBMLSolverOscilatorDemoSteppables import SBMLSolverOscilatorDemoSteppable


CompuCellSetup.register_steppable(steppable=SBMLSolverOscilatorDemoSteppable(frequency=1))
CompuCellSetup.register_compiled_steppable(
    library_path='Native/SBMLSolverOscilatorClassSteppable',
    frequency=1,
)

CompuCellSetup.run()



