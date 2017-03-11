from multiprocessing import Process
from subprocess import call
from MonitorBase import *
import zmq
import hashlib
import datetime
import os
from os.path import *

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
    def create_project_directory(self, param_dict, template_project_path, workspace_dir):
        hasher = hashlib.sha1()

    def get_formatted_timestamp(self):
        return datetime.datetime.fromtimestamp(time.time()).strftime('%d_%m_%Y_%H_%M_%S')

    def get_output_dir_name(self, simulation_name, workspace_dir):

        simulation_corename, ext = splitext(basename(simulation_name))

        if len(ext):
            ext = ext[1:]

        output_dir_name = join(workspace_dir, simulation_corename + '_' + ext + '_' + self.get_formatted_timestamp())

        return output_dir_name

    # def create_output_dir(self, dirname):
    #     if isdir(dirname):
    #         raise IOError('Could not create output directory %s. It already exist' %dirname)
    #
    #     os.makedirs(dirname)

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
        simulation_name = workload_json['simulation_filename']
        cc3d_command = workload_json['cc3d_command']
        workspace_dir = workload_json['workspace_dir']

        print 'received param_dict = ', param_dict

        simulation_output_dir = self.get_output_dir_name(simulation_name=simulation_name, workspace_dir=workspace_dir)

        # data = work['num']

        popen_args = [cc3d_command]
        # popen_args=[r'C:\CompuCell3D-64bit\runScript.bat']

        # popen_args.append("--pushAddress=%s"%self.pull_address_str)
        # simulation_name = r'D:\CC3DProjects\short_demo\short_demo.cc3d'
        output_frequency = 10
        if simulation_name != "":
            popen_args.append("-i")
            popen_args.append(simulation_name)

        # forcing run script to use custom output directory
        popen_args.append("-o")
        popen_args.append(simulation_output_dir)

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


        # print 'got data ', data, ' id =', self.id_number
        # result = {'consumer': self.id_number, 'num': data}
        # consumer_sender.send_json(result)
