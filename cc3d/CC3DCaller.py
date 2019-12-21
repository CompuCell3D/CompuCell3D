import traceback
import cc3d
import sys
from os.path import *

# import cc3d.CompuCellSetup as CompuCellSetup
from cc3d import CompuCellSetup
from cc3d.CompuCellSetup.sim_runner import run_cc3d_project
from cc3d.core.RollbackImporter import RollbackImporter


class CC3DCaller:
    def __init__(self):
        self.cc3d_sim_fname = None
        self.output_frequency = 0
        self.screenshot_output_frequency = 0
        self.restart_snapshot_frequency = 0
        self.restart_multiple_snapshots = False
        self.output_dir = None
        self.output_file_core_name = None

    def run(self):

        persistent_globals = cc3d.CompuCellSetup.persistent_globals

        rollback_importer = RollbackImporter()

        persistent_globals.simulation_file_name = self.cc3d_sim_fname
        persistent_globals.output_frequency = self.output_frequency
        persistent_globals.screenshot_output_frequency = self.screenshot_output_frequency
        persistent_globals.set_output_dir(self.output_dir)
        persistent_globals.output_file_core_name = self.output_file_core_name
        persistent_globals.restart_snapshot_frequency = self.restart_snapshot_frequency
        persistent_globals.restart_multiple_snapshots = self.restart_multiple_snapshots

        run_cc3d_project(cc3d_sim_fname=self.cc3d_sim_fname)

        rollback_importer.uninstall()

        return persistent_globals.return_object
