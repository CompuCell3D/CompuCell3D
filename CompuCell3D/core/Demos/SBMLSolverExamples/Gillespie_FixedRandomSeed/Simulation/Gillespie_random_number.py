from cc3d import CompuCellSetup        

from Gillespie_random_numberSteppables import Gillespie_random_numberSteppable
CompuCellSetup.register_steppable(steppable=Gillespie_random_numberSteppable(frequency=1))

CompuCellSetup.run()
