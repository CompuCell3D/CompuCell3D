
from cc3d import CompuCellSetup
        

from PDESuiteTest1Steppables import PDESuiteTest1Steppable

CompuCellSetup.register_steppable(steppable=PDESuiteTest1Steppable(frequency=1000))


CompuCellSetup.run()
