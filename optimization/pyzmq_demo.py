#!/usr/bin/env python

"""
Pass data between processes started through the multiprocessing module
using pyzmq and process them with PyCUDA
"""

import numpy as np
import zmq
import multiprocessing as mp
from MonitorProcessZMQ import MonitorProcessZMQ

gpu = 0


def worker():

    print 'hello'

    # context = zmq.Context()
    # socket = context.socket(zmq.REP)
    # socket.connect("tcp://localhost:5555")

    # # Process data sent to worker until a quit signal is transmitted:
    # while True:
    #     data = socket.recv_pyobj()
    #     print "Worker %i: %s" % (gpu, data)
    #     if data == 'quit':
    #         break
    #
    #     socket.send_pyobj(20.0)


def master():
    # Data to send to worker:
    data_list = map(lambda x: np.random.rand(4, 4), xrange(4))

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.bind("tcp://*:5558")

    # Send data out for processing and get back the results:
    for i in xrange(len(data_list)):
        socket.send_pyobj(data_list[i])
        result = socket.recv_pyobj()
        print "Master: ", result
    socket.send_pyobj('quit')


if __name__ == '__main__':

    worker = MonitorProcessZMQ()
    worker.start()

    master()

    # worker = mp.Process(target=worker)
    # worker.start()
    #
    # master()
