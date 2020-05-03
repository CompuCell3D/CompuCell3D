from cc3d import CompuCellSetup

from obstruction_chemotaxisSteppables import obstruction_chemotaxisSteppable

CompuCellSetup.register_steppable(steppable=obstruction_chemotaxisSteppable(frequency=1))

CompuCellSetup.run()
