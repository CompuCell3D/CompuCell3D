from .utils import *
from . simulation_utils import *
from . sim_runner import *
from . readers import *
from . simulation_setup import *
# from . readers import readCC3DFile
# from . initializers import (initializeSimulationObjects,
#                             initialize_cc3d,
#                             run,
#                             register_steppable,
#                             getCoreSimulationObjects,
#                             mainLoop)
from . persistent_globals import PersistentGlobals
# cc3dSimulationDataHandler = ''

persistent_globals = PersistentGlobals()

def resetGlobals():
    """
    Resets persisten globals
    :return:
    """
    global  persistent_globals
    persistent_globals = PersistentGlobals()
