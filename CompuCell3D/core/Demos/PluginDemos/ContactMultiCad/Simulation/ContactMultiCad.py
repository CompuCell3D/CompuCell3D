from cc3d import CompuCellSetup
from .ContactMultiCadSteppables import ContactMultiCadSteppable

cmc_steppable = ContactMultiCadSteppable(frequency=10)
cmc_steppable.set_type_contact_energy_table({0: 0.0, 1: 20, 2: 30})

CompuCellSetup.register_steppable(steppable=cmc_steppable)

CompuCellSetup.run()
