from cc3d import CompuCellSetup
from .PythonCppDataExchangeSteppables import PythonExchangeSteppable


CompuCellSetup.register_steppable(steppable=PythonExchangeSteppable(frequency=1))
CompuCellSetup.register_compiled_steppable(
    library_path="Native/PythonCppDataExchangeCompiledSteppable", frequency=1)
CompuCellSetup.run()
