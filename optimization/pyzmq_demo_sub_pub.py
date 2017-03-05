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


# def master():
#     # Data to send to worker:
#     data_list = map(lambda x: np.random.rand(4, 4), xrange(4))
#
#     context = zmq.Context()
#     socket = context.socket(zmq.REQ)
#     socket.bind("tcp://*:5558")
#
#     # Send data out for processing and get back the results:
#     for i in xrange(len(data_list)):
#         socket.send_pyobj(data_list[i])
#         result = socket.recv_pyobj()
#         print "Master: ", result
#     socket.send_pyobj('quit')

def master(num_workers=1):
    # Data to send to worker:
    data_list = map(lambda x: np.random.rand(4, 4), xrange(4))

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.bind("tcp://*:5558")

    # Send data out for processing and get back the results:
    for i in xrange(len(data_list)):

        for w in xrange(num_workers):
            socket.send_pyobj(data_list[i])
        for w in xrange(num_workers):
            result = socket.recv_pyobj()
            print "Master: ", result

        # result = socket.recv_pyobj()
        # print "Master: ", result

    for w in xrange(num_workers):
        socket.send_pyobj('quit')





if __name__ == '__main__':

    worker = MonitorProcessZMQ(id_number=0,name='worker_0')
    worker.start()

    worker1 = MonitorProcessZMQ(id_number=1,name='worker_1')
    worker1.start()


    master(2)

    # worker = mp.Process(target=worker)
    # worker.start()
    #
    # master()
