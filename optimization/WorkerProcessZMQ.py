from multiprocessing import Process
from MonitorBase import *
import zmq
import random
class WorkerProcessZMQ(MonitorBase,Process):
    def __init__(self,id_number=-1, name='generic_monitor', configuration=None, session_data=None):
        MonitorBase.__init__(self,id_number=id_number, name=name, configuration=configuration, session_data=session_data)
        Process.__init__(self)


    def initialize(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        # self.socket = self.context.socket(zmq.SUB)
        self.socket.connect("tcp://localhost:5558")

    def event_loop(self):


        consumer_id = random.randrange(1, 10005)
        print "I am consumer #%s" % (consumer_id)
        context = zmq.Context()
        # recieve work
        consumer_receiver = context.socket(zmq.PULL)
        consumer_receiver.connect("tcp://127.0.0.1:5557")
        # send work
        consumer_sender = context.socket(zmq.PUSH)
        consumer_sender.connect("tcp://127.0.0.1:5558")

        while True:
            work = consumer_receiver.recv_json()
            data = work['num']
            # print 'got data ', data, ' id =', self.id_number
            result = {'consumer': consumer_id, 'num': data}
            if data % 2 == 0:
                consumer_sender.send_json(result)



