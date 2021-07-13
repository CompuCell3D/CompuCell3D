
from cc3d import CompuCellSetup
        

from PDETestPeriodicFESteppables import PDETestPeriodicFESteppable

CompuCellSetup.register_steppable(steppable=PDETestPeriodicFESteppable(frequency=100))


CompuCellSetup.run()
