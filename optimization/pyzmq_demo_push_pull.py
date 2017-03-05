#!/usr/bin/env python

"""
Pass data between processes started through the multiprocessing module
using pyzmq and process them with PyCUDA
"""

import numpy as np
import zmq
import multiprocessing as mp
from WorkerProcessZMQ import WorkerProcessZMQ
from ReduceProcessZMQ import ReduceProcessZMQ
import random


import time
import zmq


class Optimizer(object):
    def __init__(self):
        self.param_set_list = [1,2,3,4,5,6,7,8,9,10,11,12]
        self.context = zmq.Context()
        self.zmq_socket = self.context.socket(zmq.PUSH)
        self.zmq_socket.bind("tcp://127.0.0.1:5557")
        self.num_workers = 4

    def reduce(self, num_workers):

        context = zmq.Context()
        results_receiver = context.socket(zmq.PULL)
        results_receiver.bind("tcp://127.0.0.1:5558")
        collecter_data = {}
        for x in xrange(num_workers):
            result = results_receiver.recv_json()


    def run(self):

        for param in self.param_set_list:




def producer():
    context = zmq.Context()
    zmq_socket = context.socket(zmq.PUSH)
    zmq_socket.bind("tcp://127.0.0.1:5557")
    # Start your result manager and workers before you start your producers
    for num in xrange(20000):

        work_message = { 'num' : num }
        zmq_socket.send_json(work_message)





if __name__ == '__main__':

    worker = WorkerProcessZMQ(id_number=0,name='worker_0')
    worker.start()

    worker1 = WorkerProcessZMQ(id_number=1,name='worker_1')
    worker1.start()


    reducer = ReduceProcessZMQ(id_number=2,name='worker_2')
    reducer.start()

    time.sleep(2.0)

    producer()

    # master(2)

    # worker = mp.Process(target=worker)
    # worker.start()
    #
    # master()
