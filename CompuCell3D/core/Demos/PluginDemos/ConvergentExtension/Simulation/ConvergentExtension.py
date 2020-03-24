from cc3d import CompuCellSetup

from ConvergentExtensionSteppables import ConvergentExtensionSteppable

CompuCellSetup.register_steppable(steppable=ConvergentExtensionSteppable(frequency=1))

CompuCellSetup.run()
