from cc3d import CompuCellSetup
from .scientificPlotsMultipleAxesSteppables import ExtraPlotSteppable

CompuCellSetup.register_steppable(steppable=ExtraPlotSteppable(frequency=1))

CompuCellSetup.run()

