#!/usr/bin/env python

# TODO add discovery of free port - zmw have a fcn that facilitates it (port range is given )
# port_selected = socket.bind_to_random_port('tcp://*', min_port=6001, max_port=6004, max_tries=100)
#


import numpy as np
from OptimizerWorkerProcessZMQ import OptimizerWorkerProcessZMQ
from collections import OrderedDict
import os
from os.path import *
import datetime
import time
import json
import random

import time
import zmq
import argparse
import sys
import json
import numpy as np


class OptimizationCMLParser(object):
    def __init__(self, arg_count_threshold=1):
        self.parser = argparse.ArgumentParser(description='CMA Optimization')
        self.parser.add_argument('-i', '--input', required=True, action='store')
        self.parser.add_argument('-p', '--params-file', required=True, action='store', default='')

        self.arg_list = []
        self.arg_count_threshold = arg_count_threshold

    def arg(self, name, *vals):
        self.arg_list.append(name)
        for val in vals:
            self.arg_list.append(val)

    def configure_python_paths(self, paths):
        for path in paths:
            sys.path.append(path)

    def parse(self):
        print sys.argv
        if len(sys.argv) <= self.arg_count_threshold and len(self.arg_list):
            args = self.parser.parse_args(self.arg_list)
        else:
            args = self.parser.parse_args()

        return args


class OptimizationParameterManager(object):
    def __init__(self):
        self.params_jn = None
        self.parameters = None

        self.params_names = []
        self.std_dev = None

    def read_paramters(self, fname):
        self.params_jn = json.load(open(fname, 'r'))
        self.parameters = self.params_jn['parameters']
        self.params_names = self.parameters.keys()

        self.params_bounds = np.zeros((len(self.params_names), 2), dtype=np.float)

        for i, name in enumerate(self.params_names):
            self.params_bounds[i, :] = self.parameters[name]

        vec = np.array([0.5,0.5])
        true_params = self.params_from_0_1(vec)

        true_params_dict = self.param_from_0_1_dict(vec)
        print true_params_dict

        print

    def get_starting_points(self):
        return 0.5 * np.ones(len(self.params_names), dtype=np.float)

    def params_from_0_1(self, param_vec_0_1):
        return self.params_bounds[:, 0] + param_vec_0_1 * (self.params_bounds[:, 1] - self.params_bounds[:, 0])

    def param_from_0_1_dict(self, param_vec_0_1):
        return dict(zip(self.params_names, self.params_from_0_1(param_vec_0_1)))


class Optimizer(object):
    def __init__(self):
        # self.param_set_list = [1,2,3,4,5,6,7,8,9,10,11,12,13]
        self.param_set_list = [1, 2, 3, 4, 5]
        self.context = zmq.Context()
        self.push_address_str = "tcp://127.0.0.1:5557"
        self.pull_address_str = "tcp://127.0.0.1:5558"

        self.zmq_socket = self.context.socket(zmq.PUSH)
        self.zmq_socket.bind(self.push_address_str)
        self.num_workers = 1

    def acknowledge_presence(self, num_workers):
        """
        Receives handshamke message from workers
        :param num_workers: {int} number of workers
        :return:None
        """

        context = zmq.Context()
        results_receiver = context.socket(zmq.PULL)
        results_receiver.bind(self.pull_address_str)

        for x in xrange(num_workers):
            result = results_receiver.recv_json()
            print 'worker x=', x, ' ready'

    def reduce(self, num_workers):
        """
        receives output from workers
        :param num_workers: {int} number of workers
        :return: None
        """

        context = zmq.Context()
        results_receiver = context.socket(zmq.PULL)
        results_receiver.bind(self.pull_address_str)
        collecter_data = {}
        sum = 0
        print 'reducing=', num_workers, ' workers'
        for x in xrange(num_workers):
            print 'waiting for worker x=', x
            result = results_receiver.recv_json()
            print 'got result from worker x=', x, ' res=', result
            sum += result['num']

        print 'sum = ', sum

    def param_generator(self, num_workers):
        counter = 0
        current_set = []
        for param in self.param_set_list:
            current_set.append(param)
            counter += 1
            if counter == num_workers:
                yield current_set
                current_set = []
                counter = 0

        # sending reminder of the params
        if len(current_set):
            yield current_set

    def get_formatted_timestamp(self):
        return datetime.datetime.fromtimestamp(time.time()).strftime('%d_%m_%Y_%H_%M_%S')

    def create_workspace_dir(self, simulation_corename, workspace_root_dir):
        """
        Creates workspace directory for the optimization job
        :param simulation_corename: {str} simulation file name (basename)
        :param workspace_root_dir: {str} CC3D workspace root dir (typically ~/CC3DWorkspace)
        :return: None
        """

        formatted_timestamp = self.get_formatted_timestamp()
        workspace_dir = join(workspace_root_dir, simulation_corename + '_opt_' + formatted_timestamp)

        # creating workspace directory
        if isdir(workspace_dir):
            raise IOError('Cannot create output directory %s. This directory already exist' % workspace_dir)

        os.makedirs(workspace_dir)

        return workspace_dir

    def run_task(self, workload_dict, param_set):
        """
        Dispatches simulation jobs to workers based on the parameter set (param_set)
        :param workload_dict: {dictionary-like} workload information to be sent to worker - does not include param_dict.
        param_dict will be set by this function
        :param param_set: parameter_set
        :return: None
        """

        self.worker_pool = []

        num_params = len(param_set)

        # for w in xrange(self.num_workers):
        for w in xrange(num_params):
            worker = OptimizerWorkerProcessZMQ(id_number=w, name='worker_%s' % w)
            worker.set_pull_address_str(self.push_address_str)
            worker.set_push_address_str(self.pull_address_str)
            self.worker_pool.append(worker)

        for worker in self.worker_pool:
            worker.start()

        time.sleep(1.0)

        # self.acknowledge_presence(self.num_workers)

        for param_idx, param in enumerate(param_set):
            param_labels = ['TEMPERATURE', 'CONTACT_A_B']
            param_vals = [12.2, float(param)]
            param_dict = OrderedDict(zip(param_labels, param_vals))

            # appending param_dict to  workload dict
            workload_dict['param_dict'] = param_dict

            # self.zmq_socket.send_json(param_dict)
            self.zmq_socket.send_json(workload_dict)

            print 'sent = ', workload_dict

        print 'WILL REDUCE ', param_idx + 1, ' workers'
        self.reduce(param_idx + 1)
        print 'FINISHED REDUCING'

    def prepare_optimization_run(self, simulation_name):
        """
        Prepares optimization run which includes: creating workspace directory for the optimization job
        and creating workload_dictionary (will be sent as json to worker)
        :param simulation_name: {str} simulation name
        :return: {dictionary-like} workload dictionary -  does not include param_dict
        """
        # path to cc3d project


        # constructing workspace dir for all jobs that are part of current optimization run
        workspace_root_dir = join(expanduser('~'), 'CC3DWorkspace')

        simulation_corename, ext = splitext(basename(simulation_name))

        workspace_dir = self.create_workspace_dir(simulation_corename, workspace_root_dir)

        workload_dict = OrderedDict()
        workload_dict['cc3d_command'] = r'C:\CompuCell3D-64bit\runScript.bat'
        workload_dict['workspace_dir'] = workspace_dir
        workload_dict['simulation_filename'] = simulation_name
        workload_dict['param_dict'] = None

        return workload_dict

    def run(self):

        """
        Runs optimization job
        :return:
        """
        simulation_name = r'D:\CC3DProjects\short_demo\short_demo.cc3d'
        workload_dict = self.prepare_optimization_run(simulation_name=simulation_name)

        for param_set in self.param_generator(self.num_workers):
            print 'CURRENT PARAM SET=', param_set
            self.run_task(workload_dict, param_set)
            print 'FINISHED PARAM_SET=', param_set


            # def run(self):
            #     for param_set in self.param_generator(self.num_workers):
            #         print 'CURRENT PARAM SET=', param_set
            #
            #         print 'FINISHED PARAM_SET=', param_set


if __name__ == '__main__':
    # from subprocess import call
    # popen_args = [r'C:\CompuCell3D-64bit\runScript.bat']
    #
    # # popen_args.append("--pushAddress=%s"%self.pull_address_str)
    # simulation_name = r'D:\CC3DProjects\short_demo\short_demo.cc3d'
    # output_frequency = 10
    # if simulation_name != "":
    #     popen_args.append("-i")
    #     popen_args.append(simulation_name)
    #
    # if output_frequency > 0:
    #
    #     popen_args.append("-f")
    #     popen_args.append(str(output_frequency))
    # else:
    #     popen_args.append("--noOutput")
    #
    # # popen_args.append("-p" )
    # # popen_args.append(str(self.pull_address_str))
    #
    # # popen_args.append("--pushAddress=%s" % str("dupa"))
    # popen_args.append("-p" )
    # popen_args.append("dupa")
    #
    # # popen_args.append(str(self.pull_address_str))
    #
    #
    # print 'popen_args=', popen_args
    # # sys.exit()
    # # this call will block until simulattion is done
    # call(popen_args)

    cml_parser = OptimizationCMLParser()

    cml_parser.arg('--input', r'D:\CC3DProjects\short_demo\short_demo.cc3d')
    cml_parser.arg('--params-file', r'D:\CC3D_GIT\optimization\params.json')
    args = cml_parser.parse()

    optim_param_mgr = OptimizationParameterManager()
    optim_param_mgr.read_paramters(args.params_file)

    print args.input
    print args.params_file

    # optimizer = Optimizer()
    # optimizer.run()
