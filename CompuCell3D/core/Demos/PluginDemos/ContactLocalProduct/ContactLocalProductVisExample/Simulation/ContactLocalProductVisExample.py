from cc3d import CompuCellSetup
from .ContactLocalProductExampleModules import ContactLocalProductSteppable
from .ContactLocalProductExampleModules import ContactSpecVisualizationSteppable

clp_steppable = ContactLocalProductSteppable(frequency=10)
clp_steppable.set_type_contact_energy_table({0: 0.0, 1: 20, 2: 30})

# alternative call when we select random number for adhesion molecules form the specified interval e.g. [20,30]
# clp_steppable.set_type_contact_energy_table({0:0.0 , 1:[20,30], 2:[30,50]})


CompuCellSetup.register_steppable(steppable=clp_steppable)
CompuCellSetup.register_steppable(steppable=ContactSpecVisualizationSteppable(frequency=50))

CompuCellSetup.run()


