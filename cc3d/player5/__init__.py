from cc3d.core.GraphicsOffScreen import *


def _hook_cc3d_player():
    """
    Hooks Player configuration with core: Player uses a static configuration and has additional settings

    :return: None
    """
    from . import Configuration
    from cc3d.CompuCellSetup import persistent_globals

    def get_configuration():
        return Configuration

    persistent_globals.set_configuration_getter(get_configuration)


_hook_cc3d_player()
