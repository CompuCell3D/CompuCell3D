import cc3d
from cc3d import CompuCellSetup
from cc3d.CompuCellSetup.sim_runner import run_cc3d_project
from cc3d.core.RollbackImporter import RollbackImporter
import multiprocessing


class CC3DCaller:
    def __init__(self,
                 cc3d_sim_fname=None,
                 output_frequency=0,
                 screenshot_output_frequency=0,
                 restart_snapshot_frequency=0,
                 restart_multiple_snapshots=False,
                 output_dir=None,
                 output_file_core_name=None,
                 result_identifier_tag=None,
                 sim_input=None):

        self.cc3d_sim_fname = cc3d_sim_fname
        self.output_frequency = output_frequency
        self.screenshot_output_frequency = screenshot_output_frequency
        self.restart_snapshot_frequency = restart_snapshot_frequency
        self.restart_multiple_snapshots = restart_multiple_snapshots
        self.output_dir = output_dir
        self.output_file_core_name = output_file_core_name
        self.result_identifier_tag = result_identifier_tag
        self.sim_input = sim_input

    def run(self):
        persistent_globals = cc3d.CompuCellSetup.persistent_globals

        # Clear lingering persistent global data from previous run if any
        if persistent_globals.simulation_initialized:
            simulator = persistent_globals.simulator
            simulator.cleanAfterSimulation()
            cc3d.CompuCellSetup.resetGlobals()
            persistent_globals = cc3d.CompuCellSetup.persistent_globals

        rollback_importer = RollbackImporter()

        persistent_globals.simulation_file_name = self.cc3d_sim_fname
        persistent_globals.output_frequency = self.output_frequency
        persistent_globals.screenshot_output_frequency = self.screenshot_output_frequency
        persistent_globals.set_output_dir(self.output_dir)
        persistent_globals.output_file_core_name = self.output_file_core_name
        persistent_globals.restart_snapshot_frequency = self.restart_snapshot_frequency
        persistent_globals.restart_multiple_snapshots = self.restart_multiple_snapshots
        persistent_globals.input_object = self.sim_input

        run_cc3d_project(cc3d_sim_fname=self.cc3d_sim_fname)

        rollback_importer.uninstall()

        if self.result_identifier_tag is not None:
            return {
                'tag': self.result_identifier_tag,
                'result': persistent_globals.return_object
            }
        else:
            return {'result': persistent_globals.return_object}


class CC3DCallerWorker(multiprocessing.Process):

    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):

        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                print('%s: Exiting' % proc_name)
                self.task_queue.task_done()
                break
            print('%s: %s' % (proc_name, next_task))
            answer = next_task.run()
            self.task_queue.task_done()
            self.result_queue.put(answer)
        return
