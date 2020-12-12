from cc3d import CompuCellSetup
from .ContactLocalProductExampleModules import ContactLocalProductSteppable

clp_steppable = ContactLocalProductSteppable(frequency=10)
clp_steppable.set_type_contact_energy_table({0: 0.0, 1: 20, 2: 30})

CompuCellSetup.register_steppable(steppable=clp_steppable)

CompuCellSetup.run()
