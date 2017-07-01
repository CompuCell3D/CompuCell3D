import datetime
import os
import subprocess
from os.path import *
from threading import Thread, Lock

import jinja2
import psutil

from rampy.util import mkdir_p
import fnmatch
import os
import h5py
from numpy.testing import assert_array_equal

abs_join = lambda *args: abspath(join(*args))
from rampy.emulator_utils.task_laptop_utils import TaskLaptopScriptGenerator


def find_file_in_dir(dirname, fname_pattern):
    """
    Does recursive filename match in the directory. Returns all matches found in the directory
    :param dirname: directory to search
    :param fname_pattern: file name pattern (wildcard expression)
    :return: list of matches
    """

    matches = []
    for root, dirnames, filenames in os.walk(dirname):
        for filename in fnmatch.filter(filenames, fname_pattern):
            matches.append(os.path.join(root, filename))
    return matches

def compare_eegs(dir_path, orig_eeg_file):
    """
    Compares the content of the original eeg file to the content that is saved during emulated ENS run
    :param dir_path: directory where the output of the emulated run is written to
    :param orig_eeg_file: original eeg file from the actual session
    :return: None
    """

    recorded_eeg_file_list = find_file_in_dir(dirname=dir_path, fname_pattern='*.h5')
    eeg_file = recorded_eeg_file_list[0]

    with h5py.File(eeg_file, 'r') as eeg_hfile, h5py.File(orig_eeg_file, 'r') as orig_eeg_hfile :
        eegs = eeg_hfile['timeseries'].value
        orig_eegs = orig_eeg_hfile['timeseries'].value

        num_samples_to_compare = eegs.shape[0]

        assert_array_equal(eegs[:num_samples_to_compare,:], orig_eegs[:num_samples_to_compare,:])

def check_output_codes(dir_path):
    """
    Checks if all files with .output extension have errror_code == "0"
    :param dir_path: path where we look for .output files
    :return: None
    """

    output_file_list = find_file_in_dir(dirname=dir_path, fname_pattern='*.output')

    for output_file in output_file_list:
        with open(output_file, 'r') as f:
            for line in f.readlines():
                assert line.strip() == '0'

def clean_leftover_processes(pid=None):
    if pid is None:
        pid = os.getpid()
    current_process = psutil.Process(pid)
    current_child_processes = current_process.children()

    for proc in current_child_processes:
        print ('process_name = ', proc.name())
        if proc.name().lower().startswith('python'):
            print('KILLING PYTHON PROCESS WITH PID=%s', str(proc.pid))
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                print ('Could not locate {} process pid'.format(proc.name(), proc.pid()))

def prepare_run_script(script_template_name, substitution_dict=None):
    """
    generates run script from templates. The script is placed in the same directory as the template
    :param script_template_name: run script template
    :param substitution_dict: substitution dict (jinja2) -  look for {{ var }} in the script for required entries
    :return: full path to the generated script
    """

    path_cwd = dirname(__file__)
    loader = jinja2.FileSystemLoader(path_cwd)
    jenv = jinja2.Environment(loader=loader, trim_blocks=True, lstrip_blocks=True)
    script_template = jenv.get_template(script_template_name)
    script_template_rendered = script_template.render(**substitution_dict)

    print script_template_rendered
    out_script_name = abs_join(path_cwd, splitext(script_template_name)[0])
    with open(out_script_name, 'w') as f_out:
        f_out.write('%s' % script_template_rendered)

    return out_script_name


class TestSpecs():

    # path to ramulator run script
    ramulator_bat = r'D:\src\SYS3\ramulator.bat'
    # path to ram_control run script
    ram_control_bat = r'D:\src\SYS3\ram_control.bat'
    # full path experiment_config.json to run
    experiment_config = r'd:\data_out\R1308T\FR1\experiment_config.json'
    # path to eeg_timeseries.h5 that will be used as a source of streamed eeg data
    emulated_eeg_file = r'D:\SYS3_testdata\R1308T\FR1\20170611_172330\eeg_timeseries.h5'
    # task laptop address
    task_laptop_address = r'tcp://127.0.0.1:8889'
    # .csv file that ram_control will "run" to emulate task laptop
    task_laptop_script = r''
    # top directory where the data from emulated run gets written
    test_output_root = 'D:\sys3_test_output'
    # the directory - created for each individual run where the data from emulated run gets written
    test_output_dir = ''
    subject = ''
    experiment = ''
    # list of phase types to be used in list autogeneration (e.g. ['PRACTICE','STIM ENCODING', 'STIM ENCODING',...])
    phase_types = ''
    # delay in ms of thr final exit of the ramulator - allows completion of en-of-session tasks in monitors
    exit_on_session_end_delay = 0


class TestExecutor(object):
    def __init__(self, test_specs, *args, **kwds):
        self.test_specs = test_specs

    def populate_task_laptop_script_template(self):
        """
        Task-specific -  populates messages in the task laptop script template
        Saves new task laptop scripts and modifies self.test_specs accordingly.
        Default implementation does not do anything
        :return: None
        """
        pass

    def run(self):
        """
        Runs single session - spawns ramulator and ram_control processes
        :return: None
        """
        ts = self.test_specs

        # creating test output dir
        datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ts.test_output_dir = abs_join(ts.test_output_root, ts.subject, ts.experiment, datetime_str)
        mkdir_p(ts.test_output_dir)

        self.populate_task_laptop_script_template()

        ramulator_args = [ts.ramulator_bat,
                          r'--autoplay',
                          r'--experiment-config=%s' % ts.experiment_config,
                          r'--emulated-eeg-file=%s' % ts.emulated_eeg_file,
                          r'--task-laptop-address=%s' % ts.task_laptop_address,
                          r'--exit-on-error',
                          r'--data-dir=%s' % ts.test_output_dir,
                          r'--exit-on-session-end',
                          r'--exit-on-session-end-delay=%s'%ts.exit_on_session_end_delay,
                          ]

        ram_control_args = [ts.ram_control_bat,
                            '-f',
                            r'%s' % ts.task_laptop_script
                            ]

        ram_control_runner = CMLRunner(args=ram_control_args)

        ramulator_runner = CMLRunner(args=ramulator_args, output_dir=ts.test_output_dir, kill_dependents_flag=True)
        ramulator_runner.add_dependent_runner(ram_control_runner)

        ramulator_runner.start()
        ram_control_runner.start()

        ramulator_runner.join()
        ram_control_runner.join()

        clean_leftover_processes()

    def post_run(self):
        ts = self.test_specs
        check_output_codes(dir_path=ts.test_output_dir)
        compare_eegs(dir_path=ts.test_output_dir, orig_eeg_file=ts.emulated_eeg_file)


class TestExecutorAutogen(TestExecutor):
    def __init__(self, *args, **kwds):
        super(TestExecutorAutogen, self).__init__(*args, **kwds)

    def populate_task_laptop_script_template(self):
        """
        Task-specific -  populates messages in the task laptop script template
        Saves new task laptop scripts and modifies self.test_specs accordingly.
        Default implementation does not do anything
        :return: None
        """
        # when we specify task_laptop_script we skip automatic script generation
        if self.test_specs.task_laptop_script:
            return

        ts = self.test_specs


        templates_dict = {}


        templates_dict['fr'] = {'main_template': 'templates/fr_main.j2',
                             'list_template': 'templates/fr_list.j2',
                             'retrieval_template': 'templates/fr_retrieval.j2'
                             }

        templates_dict['catfr'] = templates_dict['fr']
        templates_dict['ps4_fr'] = templates_dict['fr']
        templates_dict['ps4_catfr'] = templates_dict['fr']

        templates_dict['pal'] = {'main_template': 'templates/pal_main.j2',
                             'list_template': 'templates/pal_list.j2',
                             'retrieval_template': 'templates/pal_retrieval.j2'
                             }

        templates_dict['ps4_pal'] = templates_dict['pal']


        exp_template_dict = None

        for exp_code in  templates_dict.keys():
            if ts.experiment.lower().startswith(exp_code):
                exp_template_dict = templates_dict[exp_code]
                break

        if exp_template_dict is None:
            raise KeyError('Could not find templates for {}. TestExecutorAutogen cannot continue'.format(ts.experiment))


        tl_script_gen = TaskLaptopScriptGenerator(**exp_template_dict)

        main_template_rendered = tl_script_gen.generate_task_laptop_script(phase_types=ts.phase_types,
                                                                           subject=ts.subject,
                                                                           experiment=ts.experiment)

        ram_control_csv_script = abspath(join(ts.test_output_dir, '{}_main.csv'.format(ts.experiment)))
        with open(ram_control_csv_script, 'w') as f:
            f.write('%s' % main_template_rendered)

        # pointing to the just-written ram_control script
        ts.task_laptop_script = ram_control_csv_script


class CMLRunner(Thread):
    def __init__(self, args, output_dir=None, kill_dependents_flag=False):
        Thread.__init__(self)
        self.args = args
        self.process_handle = None
        self.process_pid = None
        self.process_error_code = None
        self.kill_dependents_flag = kill_dependents_flag
        self.dependents = []
        self.output_dir = output_dir
        self.lock = Lock()

    def add_dependent_runner(self, dependent_runner):
        self.dependents.append(dependent_runner)

    def write_report(self):

        if self.output_dir:
            fname = abs_join(self.output_dir, basename(self.args[0]) + '.output')
            with open(fname, 'w') as f:
                f.write('%s' % self.process_error_code)

    def cleanup(self):
        with self.lock:
            self.process_handle = None
            self.process_pid = None
            self.process_error_code = None

    def run(self):

        self.process_handle = subprocess.Popen(self.args)
        self.process_pid = self.process_handle.pid
        text = self.process_handle.communicate()[0]

        self.process_error_code = self.process_handle.returncode
        print ('{} self.process_error_code={}'.format(self.args[0], self.process_error_code))

        print ('finished running %s' % self.args[0])
        if self.kill_dependents_flag:
            self.kill_dependents()

        self.write_report()
        self.cleanup()

    def kill(self):
        with self.lock:
            print('got kill request for %s' % self.args[0])
            if self.process_handle is not None:
                print('got kill request for %s' % self.args[0])
                print('KILL : self.process_pid=', self.process_pid)
                clean_leftover_processes(self.process_pid)
                self.process_handle.kill()

    def kill_dependents(self):

        for dep in self.dependents:
            dep.kill()


if __name__ == '__main__':
    pass

