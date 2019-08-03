from cc3d import CompuCellSetup
from .GrowthSteppablePythonModules import GrowthSteppablePython

CompuCellSetup.register_steppable(steppable=GrowthSteppablePython(frequency=1))

CompuCellSetup.run()
