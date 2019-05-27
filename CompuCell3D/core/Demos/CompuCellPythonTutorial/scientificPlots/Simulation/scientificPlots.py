from cc3d import CompuCellSetup
from .scientificPlotsSteppables import ExtraPlotSteppable
from .scientificPlotsSteppables import ExtraMultiPlotSteppable

CompuCellSetup.register_steppable(steppable=ExtraPlotSteppable(frequency=10))
CompuCellSetup.register_steppable(steppable=ExtraMultiPlotSteppable(frequency=10))

CompuCellSetup.run()

