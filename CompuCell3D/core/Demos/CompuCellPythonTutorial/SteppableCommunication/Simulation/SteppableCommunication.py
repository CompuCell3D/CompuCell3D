
import sys
from os import environ
from os import getcwd
import string

sys.path.append(environ["PYTHON_MODULE_PATH"])


import CompuCellSetup


sim,simthread = CompuCellSetup.getCoreSimulationObjects()
            
# add extra attributes here
        
pyAttributeDictionaryAdder,dictAdder=CompuCellSetup.attachDictionaryToCells(sim)
            
CompuCellSetup.initializeSimulationObjects(sim,simthread)
# Definitions of additional Python-managed fields go here
        
#Add Python steppables here
steppableRegistry=CompuCellSetup.getSteppableRegistry()
        
from SteppableCommunicationSteppables import SteppableCommunicationSteppable
steppableInstance=SteppableCommunicationSteppable(sim,_frequency=1)
steppableRegistry.registerSteppable(steppableInstance)
        

from SteppableCommunicationSteppables import ExtraSteppable
instanceOfExtraSteppable=ExtraSteppable(_simulator=sim,_frequency=1)
steppableRegistry.registerSteppable(instanceOfExtraSteppable)

CompuCellSetup.mainLoop(sim,simthread,steppableRegistry)
        
        