
from cc3d import CompuCellSetup
        

from PDETestFlatEndFESteppables import PDETestFlatEndFESteppable

CompuCellSetup.register_steppable(steppable=PDETestFlatEndFESteppable(frequency=100))


CompuCellSetup.run()
