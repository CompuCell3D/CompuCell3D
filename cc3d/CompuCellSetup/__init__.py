from .utils import *
from . simulation_utils import *
from . sim_runner import *
from . readers import *
from . simulation_setup import *
from . simulation_player_utils import *
from . persistent_globals import PersistentGlobals

#: :class:`cc3d.CompuCellSetup.persistent_globals` instance
persistent_globals = PersistentGlobals()


def resetGlobals() -> None:
    """
    Resets persisten globals
    :return:
    """
    global persistent_globals
    persistent_globals = PersistentGlobals()
