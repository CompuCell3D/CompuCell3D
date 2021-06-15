from cc3d import CompuCellSetup

from CompartmentExampleSteppables import CompartmentExampleSteppable

CompuCellSetup.register_steppable(steppable=CompartmentExampleSteppable(frequency=1))

CompuCellSetup.run()
