from cc3d import CompuCellSetup
from vector_field_polarization_steppables import VectorFieldSteppable

CompuCellSetup.register_steppable(steppable=VectorFieldSteppable(frequency=1))

CompuCellSetup.run()
