from cc3d import CompuCellSetup

from oriented_cellsortSteppables import oriented_cellsortSteppable

CompuCellSetup.register_steppable(steppable=oriented_cellsortSteppable(frequency=1))

CompuCellSetup.run()
