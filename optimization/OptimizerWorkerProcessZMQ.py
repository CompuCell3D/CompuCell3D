from multiprocessing import Process
from MonitorBase import *
import zmq
import random
import time

class OptimizerWorkerProcessZMQ(MonitorBase,Process):
    def __init__(self,id_number=-1, name='generic_monitor', configuration=None, session_data=None):
        MonitorBase.__init__(self,id_number=id_number, name=name, configuration=configuration, session_data=session_data)
        Process.__init__(self)
        self.push_address_str = None
        self.pull_address_str = None

    def set_pull_address_str(self, address_str):
        self.pull_address_str = address_str

    def set_push_address_str(self, address_str):
        self.push_address_str = address_str


    def initialize(self):
        pass

    def event_loop(self):


        print 'I am consumer #%s' % self.id_number
        context = zmq.Context()

        # recieve work
        consumer_receiver = context.socket(zmq.PULL)
        consumer_receiver.connect(self.pull_address_str)

        # send work
        consumer_sender = context.socket(zmq.PUSH)
        consumer_sender.connect(self.push_address_str)

        # result = {'consumer': consumer_id,'OK':True}
        # consumer_sender.send_json(result)



        time.sleep(2.0)
        work = consumer_receiver.recv_json()
        data = work['num']

        print 'got data ', data, ' id =', self.id_number
        result = {'consumer': self.id_number, 'num': data}
        consumer_sender.send_json(result)



