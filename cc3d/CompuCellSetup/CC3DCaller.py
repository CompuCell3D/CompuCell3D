import cc3d
from cc3d import CompuCellSetup
from cc3d.CompuCellSetup.sim_runner import run_cc3d_project
from cc3d.core.RollbackImporter import RollbackImporter
from cc3d.CompuCellSetup.CC3DPy import CC3DPySim
import multiprocessing
# Placing here as an extended usage of this module
from cc3d.core.sim_service.impl.CC3D.CC3DSimService import CC3DSimService


class CC3DCaller(CC3DPySim):
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

        super().__init__(cc3d_sim_fname=cc3d_sim_fname,
                         output_frequency=output_frequency,
                         screenshot_output_frequency=screenshot_output_frequency,
                         restart_snapshot_frequency=restart_snapshot_frequency,
                         restart_multiple_snapshots=restart_multiple_snapshots,
                         output_dir=output_dir,
                         output_file_core_name=output_file_core_name,
                         sim_input=sim_input)

        self.result_identifier_tag = result_identifier_tag

    def run(self):
        self.init_simulation()

        rollback_importer = RollbackImporter()

        run_cc3d_project(cc3d_sim_fname=self.cc3d_sim_fname)

        rollback_importer.uninstall()

        persistent_globals = cc3d.CompuCellSetup.persistent_globals

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
