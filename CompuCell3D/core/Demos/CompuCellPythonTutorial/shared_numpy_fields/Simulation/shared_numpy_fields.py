from cc3d import CompuCellSetup
from .shared_numpy_fields_steppables import SharedNUmpyFieldsSteppable

CompuCellSetup.register_steppable(steppable=SharedNUmpyFieldsSteppable(frequency=1))

CompuCellSetup.run()
