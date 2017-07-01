import datetime
from common import *
from CMLRunner import CMLRunner


class RunExecutor(object):
    def __init__(self, run_specs):
        self.run_specs = run_specs

    # def populate_task_laptop_script_template(self):
    #     """
    #     Task-specific -  populates messages in the task laptop script template
    #     Saves new task laptop scripts and modifies self.test_specs accordingly.
    #     Default implementation does not do anything
    #     :return: None
    #     """
    #     pass

    def run(self):
        """
        Runs single simulation - spawns process
        :return: None
        """
        rs = self.run_specs

        # creating test output dir
        datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        rs.test_output_dir = abs_join(rs.test_output_root, datetime_str)
        mkdir_p(rs.test_output_dir)

        # self.populate_task_laptop_script_template()

        cc3d_args = [rs.run_command,
                          r'--input=%s'%rs.cc3d_project,
                          r'--exitWhenDone',
                          ]


        command_runner = CMLRunner(args=cc3d_args, output_dir=rs.test_output_dir, kill_dependents_flag=False)
        command_runner.add_dependent_runner(command_runner)

        command_runner.start()

        command_runner.join()

        clean_leftover_processes()

    def post_run(self):
        pass
        # ts = self.test_specs
        # check_output_codes(dir_path=ts.test_output_dir)
        # compare_eegs(dir_path=ts.test_output_dir, orig_eeg_file=ts.emulated_eeg_file)
