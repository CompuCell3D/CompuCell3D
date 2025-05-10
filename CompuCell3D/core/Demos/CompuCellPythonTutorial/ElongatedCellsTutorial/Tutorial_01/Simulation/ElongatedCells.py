from cc3d import CompuCellSetup

from ElongatedCellsSteppables import ElongatedCellsSteppable

CompuCellSetup.register_steppable(steppable=ElongatedCellsSteppable(frequency=1))

CompuCellSetup.run()
