
from cc3d import CompuCellSetup
        

from EnergyReportSteppables import EnergyReportSteppable

CompuCellSetup.register_steppable(steppable=EnergyReportSteppable(frequency=1))


CompuCellSetup.run()
