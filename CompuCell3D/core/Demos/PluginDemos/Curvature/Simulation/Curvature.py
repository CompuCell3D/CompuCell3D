from cc3d import CompuCellSetup

from CurvatureSteppables import CurvatureSteppable

CompuCellSetup.register_steppable(steppable=CurvatureSteppable(frequency=1))

CompuCellSetup.run()
