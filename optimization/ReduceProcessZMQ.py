from multiprocessing import Process
from MonitorBase import *
import zmq
import random
import pprint
class ReduceProcessZMQ(MonitorBase,Process):
    def __init__(self,id_number=-1, name='generic_monitor', configuration=None, session_data=None):
        MonitorBase.__init__(self,id_number=id_number, name=name, configuration=configuration, session_data=session_data)
        Process.__init__(self)


    def initialize(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        # self.socket = self.context.socket(zmq.SUB)
        self.socket.connect("tcp://localhost:5558")

    def event_loop(self):

        context = zmq.Context()
        results_receiver = context.socket(zmq.PULL)
        results_receiver.bind("tcp://127.0.0.1:5558")
        collecter_data = {}
        for x in xrange(1000):
            result = results_receiver.recv_json()
            # print 'result=', result
            if collecter_data.has_key(result['consumer']):
                collecter_data[result['consumer']] = collecter_data[result['consumer']] + 1
            else:
                collecter_data[result['consumer']] = 1
            if x == 999:
                pprint.pprint(collecter_data)



