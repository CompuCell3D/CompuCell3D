from threading import Thread
from common import *
from os.path import *
from threading import Lock
import subprocess





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

    def get_run_status(self):
        if int(self.process_error_code) != 0:
            return 'ERROR'
        else:
            return None

    def add_dependent_runner(self, dependent_runner):
        self.dependents.append(dependent_runner)

    def write_report(self):

        if int(self.process_error_code) != 0:
            if self.output_dir:
                # fname = abs_join(self.output_dir, basename(self.args[0]) + '.output')
                mkdir_p(self.output_dir)
                fname = abs_join(self.output_dir, 'status.output')
                with open(fname, 'w') as f:
                    f.write('%s' % self.process_error_code)

    def cleanup(self):
        with self.lock:
            self.process_handle = None
            self.process_pid = None
            # self.process_error_code = None


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
