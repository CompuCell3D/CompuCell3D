from cc3d import CompuCellSetup
from .SharedNumpyFieldsSteppables import SharedNumpyFieldsSteppable


CompuCellSetup.register_steppable(steppable=SharedNumpyFieldsSteppable(frequency=1))
CompuCellSetup.register_compiled_steppable(
    library_path="Native/SharedNumpyFieldsCompiledClassSteppable", frequency=1)
CompuCellSetup.run()
