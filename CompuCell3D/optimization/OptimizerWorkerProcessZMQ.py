from multiprocessing import Process
# from subprocess import call
import subprocess
from template_utils import generate_simulation_files_from_template
from jinja2 import Environment, FileSystemLoader
from glob import glob
from MonitorBase import *
import zmq
import hashlib
import datetime
import os
from os.path import *
import shutil

import random
import time


class OptimizerWorkerProcessZMQ(MonitorBase, Process):
    def __init__(self, id_number=-1, name='generic_monitor', configuration=None, session_data=None):
        MonitorBase.__init__(self, id_number=id_number, name=name, configuration=configuration,
                             session_data=session_data)
        Process.__init__(self)
        self.push_address_str = None
        self.pull_address_str = None

    def set_pull_address_str(self, address_str):
        """
        Sets address for the listening socket (zmq.PULL)
        :param address_str: {str} listening socket address
        :return: None
        """
        self.pull_address_str = address_str

    def set_push_address_str(self, address_str):
        """
        Sets address for the sending socket (zmq.PUSH)
        :param address_str: {str} sending socket address
        :return: None
        """


        self.push_address_str = address_str

    def initialize(self):
        """
        Not implemented
        :return:
        """
        pass


    def create_dir_hash(self, simulation_name, param_dict):
        """
        creates hashed name for the workspace directory based on  simulation_name, current_timestamp and
        string representation of parameter dictionary
        :param simulation_name:{str} full path to the simulation template
        :param param_dict: {dict} dictionary of parameters
        :return: {str} hashed corename of workspace directory
        """

        hasher = hashlib.sha1()
        hasher.update(str(param_dict) + simulation_name)

        return self.get_formatted_timestamp() + '_' + hasher.hexdigest()

    def get_formatted_timestamp(self):
        """
        Returns a string with formatted current timestamp
        :return: formatted current timestamp
        """
        return datetime.datetime.fromtimestamp(time.time()).strftime('%d_%m_%Y_%H_%M_%S')


    def create_dir(self, dirname):
        """
        Creates directory
        :param dirname:
        :return: None
        """
        if isdir(dirname):
            raise IOError('Could not create directory %s. It already exist' % dirname)

        os.makedirs(dirname)


    def generate_simulation_files_from_template(self, workspace_dir, simulation_template_name, param_dict):
        """
        Uses jinja2 templating engine to generate actual simulation files from simulation templates
        ( we use jinja2 templating syntax)
        :param workspace_dir: output directory for the current simulation
        :param simulation_template_name: full path to current cc3d simulation template - a regular cc3d simulation with
        numbers replaced by template labels
        :param param_dict: {dict} - dictionary of template parameters used to replace template labels with actual parameters
        :return : ({str},{str}) - tuple  where first element is a path to cc3d simulation generated using param_dict.
        The simulation is placed in the "hashed" directory and the second element is the "hashed" workspace dir
        """

    # dir core path
        hashed_workspace_dir_corename = self.create_dir_hash(simulation_name=simulation_template_name,
                                                             param_dict=param_dict)

        # hashed workspace dir
        hashed_workspace_dir = join(workspace_dir, hashed_workspace_dir_corename)

        simulation_dirname = join(hashed_workspace_dir, 'simulation_template')

        generated_simulation_fname, simulation_dirname = generate_simulation_files_from_template(
            simulation_dirname=simulation_dirname,
            # simulation_dirname=hashed_workspace_dir,
            simulation_template_name=simulation_template_name,param_dict=param_dict)

        return generated_simulation_fname, hashed_workspace_dir

    def cleanup_actions(self,clean_workdirs,simulation_fname):

        # removing temporary directory where we generated simulation from the simulation templates
        shutil.rmtree(dirname(simulation_fname))

        # removing workspace of the current simulation
        current_simulation_workspace_dir = dirname(dirname(simulation_fname))
        if clean_workdirs:
            shutil.rmtree(current_simulation_workspace_dir)

    def send_abort_message(self,push_address, worker_tag):
        """
        Used in case simulation throws an exception . IN this case we are sending abort message to
        optimization runner (Optimizer)
        :return: None
        """
        context = zmq.Context()


        consumer_sender = context.socket(zmq.PUSH)
        consumer_sender.connect(push_address)

        result = {'return_value_tag': worker_tag, 'return_value': -1,'abort':True}


        consumer_sender.send_json(result)


    def event_loop(self):
        """
        Main function of the worker - listens to incoming connection to retrieve workload json file, executes simulation
        and returns the result
        :return: None
        """

        print 'I am consumer #%s' % self.id_number
        context = zmq.Context()

        # recieve work
        consumer_receiver = context.socket(zmq.PULL)
        consumer_receiver.connect(self.pull_address_str)

        workload_json = consumer_receiver.recv_json()

        param_dict = workload_json['param_dict']
        simulation_template_name = workload_json['simulation_filename']
        cc3d_command = workload_json['cc3d_command']
        workspace_dir = workload_json['workspace_dir']
        worker_tag = workload_json['worker_tag']
        clean_workdirs = workload_json['clean_workdirs']

        print 'received param_dict = ', param_dict

        simulation_fname, hashed_workspace_dir = self.generate_simulation_files_from_template(
            workspace_dir=workspace_dir,
            simulation_template_name=simulation_template_name,
            param_dict=param_dict)

        popen_args = [cc3d_command]

        # output_frequency = 10
        output_frequency = -1
        if simulation_fname != "":
            popen_args.append("-i")
            popen_args.append(simulation_fname)

        # forcing run script to use custom output directory
        popen_args.append("-o")
        popen_args.append(hashed_workspace_dir)

        if output_frequency > 0:

            popen_args.append("-f")
            popen_args.append(str(output_frequency))
        else:
            popen_args.append("--noOutput")

        popen_args.append("-p")
        # popen_args.append('--pushAddress')
        popen_args.append(self.push_address_str)

        # sets return value tag
        popen_args.append("-l")
        # popen_args.append('--returnValueTag')

        popen_args.append(str(worker_tag))

        print 'popen_args=', popen_args

        # # this call will block until simulation is done
        # subprocess.popen(popen_args)
        # subprocess.call(popen_args) # this allows debugging because all the output goes to stdout

        # # this call will block until simulation is done
        try:
            # this runs single cc3d job and catches exceptions
            subprocess.check_output(popen_args)
        except subprocess.CalledProcessError as e:
            print 'GOT subprocess.CalledProcessError '
            print e.output

            self.send_abort_message(push_address=self.push_address_str, worker_tag=worker_tag)


        self.cleanup_actions(clean_workdirs=clean_workdirs,simulation_fname=simulation_fname)
