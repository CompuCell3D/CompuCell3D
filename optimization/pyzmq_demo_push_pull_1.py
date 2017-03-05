#!/usr/bin/env python

"""
Pass data between processes started through the multiprocessing module
using pyzmq and process them with PyCUDA
"""

import numpy as np
from OptimizerWorkerProcessZMQ import OptimizerWorkerProcessZMQ
import random


import time
import zmq


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

        context = zmq.Context()
        results_receiver = context.socket(zmq.PULL)
        results_receiver.bind(self.pull_address_str)


        for x in xrange(num_workers):
            result = results_receiver.recv_json()
            print 'worker x=',x,' ready'



    def reduce(self, num_workers):

        context = zmq.Context()
        results_receiver = context.socket(zmq.PULL)
        results_receiver.bind(self.pull_address_str)
        collecter_data = {}
        sum = 0
        print 'reducing=', num_workers,' workers'
        for x in xrange(num_workers):
            print 'waiting for worker x=',x
            result = results_receiver.recv_json()
            print 'got result from worker x=',x,' res=',result
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


    def run_task(self, param_set):

        self.worker_pool = []

        num_params = len(param_set)

        # for w in xrange(self.num_workers):
        for w in xrange(num_params):
            worker = OptimizerWorkerProcessZMQ(id_number=w, name='worker_%s'%w)
            worker.set_pull_address_str(self.push_address_str)
            worker.set_push_address_str(self.pull_address_str)
            self.worker_pool.append(worker)

        for worker in self.worker_pool:
            worker.start()

        time.sleep(1.0)

        # self.acknowledge_presence(self.num_workers)


        # for i in xrange(self.num_workers):
        #     num = self.param_set_list[i]
        for param_idx, param in enumerate(param_set):
            num = param

            work_message = {'num': num}

            self.zmq_socket.send_json(work_message)
            print 'sent = ',work_message

        print 'WILL REDUCE ', param_idx+1, ' workers'
        self.reduce(param_idx+1)
        print 'FINISHED REDUCING'


    def run(self):
        for param_set in self.param_generator(self.num_workers):
            print 'CURRENT PARAM SET=', param_set
            self.run_task(param_set)
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

    optimizer = Optimizer()
    optimizer.run()


