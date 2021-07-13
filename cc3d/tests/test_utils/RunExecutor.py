import datetime
from cc3d.tests.test_utils.common import *
from cc3d.tests.test_utils.CMLRunner import CMLRunner


class RunExecutor(object):
    def __init__(self, run_specs):
        self.run_specs = run_specs
        self.run_status = None

    def get_run_status(self):
        return self.run_status

    def run(self):
        """
        Runs single simulation - spawns process
        :return: None
        """
        rs = self.run_specs

        # creating test output dir
        datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        test_output_dir_rel = rs.test_output_dir if rs.test_output_dir else datetime_str
        test_output_dir_abs = abs_join(rs.test_output_root, test_output_dir_rel)

        num_steps_arg = ''
        exit_when_done_arg = ''
        if rs.player_interactive_flag:
            num_steps_arg = '--numSteps=%s' % rs.num_steps if rs.num_steps > 0 else ''
            exit_when_done_arg = r'--exitWhenDone'

        if not isinstance(rs.run_command, list):
            rs.run_command = [rs.run_command]

        cc3d_args = rs.run_command + [
                     r'--input=%s' % rs.cc3d_project,
                     exit_when_done_arg,
                     num_steps_arg
                     ]

        # removing empty args
        cc3d_args = [arg for arg in cc3d_args if arg.strip()]

        # command_runner = CMLRunner(args=cc3d_args, output_dir=rs.test_output_dir, kill_dependents_flag=False)
        command_runner = CMLRunner(args=cc3d_args, output_dir=test_output_dir_abs, kill_dependents_flag=False)
        command_runner.add_dependent_runner(command_runner)

        command_runner.start()

        command_runner.join()
        self.run_status = command_runner.get_run_status()

        clean_leftover_processes()

    def post_run(self):
        pass
