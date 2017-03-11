from multiprocessing import Process
from subprocess import call
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
        self.pull_address_str = address_str

    def set_push_address_str(self, address_str):
        self.push_address_str = address_str

    def initialize(self):
        pass

    # def event_loop(self):
    #
    #
    #     print 'I am consumer #%s' % self.id_number
    #     context = zmq.Context()
    #
    #     # recieve work
    #     consumer_receiver = context.socket(zmq.PULL)
    #     consumer_receiver.connect(self.pull_address_str)
    #
    #     # send work
    #     consumer_sender = context.socket(zmq.PUSH)
    #     consumer_sender.connect(self.push_address_str)
    #
    #     # result = {'consumer': consumer_id,'OK':True}
    #     # consumer_sender.send_json(result)
    #
    #
    #
    #     # time.sleep(2.0)
    #     work = consumer_receiver.recv_json()
    #     data = work['num']
    #
    #     print 'got data ', data, ' id =', self.id_number
    #     result = {'consumer': self.id_number, 'num': data}
    #     consumer_sender.send_json(result)
    #
    #
    #

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
        return datetime.datetime.fromtimestamp(time.time()).strftime('%d_%m_%Y_%H_%M_%S')

    def get_output_dir_name(self, simulation_template_name, workspace_dir):

        simulation_corename, ext = splitext(basename(simulation_template_name))

        if len(ext):
            ext = ext[1:]

        output_dir_name = join(workspace_dir, simulation_corename + '_' + ext + '_' + self.get_formatted_timestamp())

        return output_dir_name

    def create_dir(self, dirname):
        if isdir(dirname):
            raise IOError('Could not create directory %s. It already exist' % dirname)

        os.makedirs(dirname)

    def generate_simulation_files_from_template(self, workspace_dir, simulation_template_name, param_dict):
        """
        Used using jijja2 templating engine to generates actual simulation files from simulation templates
        (that use jinja2 templating syntax)
        :param workspace_dir: output directory for the current simulation
        :param simulation_template_name: full path to curtrent cc3d simulatotion template - a regular cc3d simulaiton with
        numbers replaced by template labels
        :param param_dict: {dict} - dictionary of template parameters used to replace template labels with actual parameters
        :return : ({str},{str}) - tuple  where first element is a path to cc3d simulation generated using param_dict. The simulation is placed
        in the "hashed" directory and the second element is the "hashed" workspace dir
        """

        # dir core path
        hashed_workspace_dir_corename = self.create_dir_hash(simulation_name=simulation_template_name,
                                                             param_dict=param_dict)

        # hashed workspace dir
        hashed_workspace_dir = join(workspace_dir, hashed_workspace_dir_corename)
        # absolute path
        tmp_simulation_template_dir = join(hashed_workspace_dir, 'simulation_template')

        # self.create_dir(tmp_simulation_template_dir)

        simulation_dir_path = dirname(simulation_template_name)
        simulation_corename = basename(simulation_template_name)

        # copying simulation dir to "hashed" directory
        shutil.copytree(src=simulation_dir_path, dst=tmp_simulation_template_dir)

        replacement_candidate_globs = ['*.py', '*xml']
        simulation_templates_path = join(tmp_simulation_template_dir, 'Simulation')
        generated_simulation_fname = join(tmp_simulation_template_dir, simulation_corename)

        replacement_candidates = []
        for glob_pattern in replacement_candidate_globs:
            replacement_candidates.extend(glob(simulation_templates_path + '/' + glob_pattern))

        j2_env = Environment(loader=FileSystemLoader(simulation_templates_path),
                             trim_blocks=True)

        for replacement_candidate_fname in replacement_candidates:
            filled_out_template_str = j2_env.get_template(basename(replacement_candidate_fname)).render(**param_dict)
            with open(replacement_candidate_fname, 'w') as fout:
                fout.write(filled_out_template_str)

        return generated_simulation_fname, hashed_workspace_dir

    def event_loop(self):

        print 'I am consumer #%s' % self.id_number
        context = zmq.Context()

        # recieve work
        consumer_receiver = context.socket(zmq.PULL)
        consumer_receiver.connect(self.pull_address_str)

        # # send work
        # consumer_sender = context.socket(zmq.PUSH)
        # consumer_sender.connect(self.push_address_str)

        # result = {'consumer': consumer_id,'OK':True}
        # consumer_sender.send_json(result)

        # time.sleep(2.0)
        workload_json = consumer_receiver.recv_json()

        param_dict = workload_json['param_dict']
        simulation_template_name = workload_json['simulation_filename']
        cc3d_command = workload_json['cc3d_command']
        workspace_dir = workload_json['workspace_dir']

        print 'received param_dict = ', param_dict

        # simulation_output_dir = self.get_output_dir_name(simulation_template_name=simulation_template_name,
        #                                                  workspace_dir=workspace_dir)
        # simulation_output_dir = workspace_dir

        simulation_fname, hashed_workspace_dir = self.generate_simulation_files_from_template(
            workspace_dir=workspace_dir,
            simulation_template_name=simulation_template_name,
            param_dict=param_dict)

        # data = work['num']

        popen_args = [cc3d_command]
        # popen_args=[r'C:\CompuCell3D-64bit\runScript.bat']

        # popen_args.append("--pushAddress=%s"%self.pull_address_str)
        # simulation_fname = r'D:\CC3DProjects\short_demo\short_demo.cc3d'
        output_frequency = 10
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
        popen_args.append(self.push_address_str)

        print 'popen_args=', popen_args

        # this call will block until simulattion is done
        call(popen_args)

        # removing temporary directory where we generated simulation from the simulation templates

        shutil.rmtree(dirname(simulation_fname))


        # print 'got data ', data, ' id =', self.id_number
        # result = {'consumer': self.id_number, 'num': data}
        # consumer_sender.send_json(result)
