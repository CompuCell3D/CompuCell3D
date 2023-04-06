# importing maboss here to avoid crashes on exit on windows... happens with latest maboss with 3.7 and swig 4
from cc3d.core.MaBoSSCC3D import MaBoSSHelper

from .utils import *
from . simulation_utils import *
from . sim_runner import *
from . readers import *
from . simulation_setup import *
from . simulation_player_utils import *
from . persistent_globals import PersistentGlobals

#: :class:`cc3d.core.CC3DSimulationDataHandler.CC3DSimulationDataHandler` instance, or None
cc3dSimulationDataHandler = None

#: :class:`cc3d.CompuCellSetup.persistent_globals` instance
persistent_globals = PersistentGlobals()


def resetGlobals() -> None:
    """
    Resets persisten globals
    :return:
    """
    global persistent_globals
    persistent_globals = PersistentGlobals()
