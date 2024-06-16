"""
Supporting infrastructure for running CC3D core in a simservice environment
"""
from cc3d.core.enums import SimType
from cc3d.CompuCellSetup.SimulationThread import SimulationThread


class SimulationServiceThread(SimulationThread):
    """Simulation thread for running as a service"""

    sim_type = SimType.SERVICE

    def __init__(self):
        super().__init__()

    @staticmethod
    def main_loop():
        from cc3d.CompuCellSetup import initialize_cc3d_sim

        def main_loop_service(sim, simthread=None, steppable_registry=None):
            initialize_cc3d_sim(sim, simthread)

        return main_loop_service

    def get_field_storage(self):
        from cc3d.CompuCellSetup import persistent_globals as pg

        return pg.persistent_holder['field_storage']
