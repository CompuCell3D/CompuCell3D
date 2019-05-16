from cc3d import CompuCellSetup
from .scientificHistBarPlotsSteppables import HistPlotSteppable
from .scientificHistBarPlotsSteppables import BarPlotSteppable

CompuCellSetup.register_steppable(steppable=HistPlotSteppable(frequency=1))
CompuCellSetup.register_steppable(steppable=BarPlotSteppable(frequency=1))

CompuCellSetup.run()
