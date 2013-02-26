import sys
from os import environ
import string
sys.path.append(environ["PYTHON_MODULE_PATH"])

import CompuCellSetup

sim,simthread = CompuCellSetup.getCoreSimulationObjects()

#Create extra player fields here or add attributes or Python plugins
CompuCellSetup.initializeSimulationObjects(sim,simthread)


#Add Python steppables here
steppableRegistry=CompuCellSetup.getSteppableRegistry()

from ContactLocalProductExampleModules import ContactLocalProductSteppable
clpSteppable=ContactLocalProductSteppable(sim)
typeContactEnergyTable={0:0.0 , 1:20, 2:30} # the format is as follows:
                                                      #type:N e.g. 1:20.1234 , 2:12.19	
#typeContactEnergyTable={0:0.0 , 1:[20,30], 2:[30,50]} # the format is as follows:
                                                      #type:[N_min,N_max] e.g. 1:[20,30] , 2:[40,50]	
clpSteppable.setTypeContactEnergyTable(typeContactEnergyTable)
steppableRegistry.registerSteppable(clpSteppable)


from ContactLocalProductExampleModules import ContactSpecVisualizationSteppable
contactVisSteppable=ContactSpecVisualizationSteppable(_simulator=sim,_frequency=50) #Here you would change frequency with which Python based visualizer is called
steppableRegistry.registerSteppable(contactVisSteppable)


CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)



