from multiprocessing import Process
from MonitorBase import *
import zmq

class MonitorProcessZMQ(MonitorBase,Process):
    def __init__(self,id_number=-1, name='generic_monitor', configuration=None, session_data=None):
        MonitorBase.__init__(self,id_number=id_number, name=name, configuration=configuration, session_data=session_data)
        Process.__init__(self)


    def initialize(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        # self.socket = self.context.socket(zmq.SUB)
        self.socket.connect("tcp://localhost:5558")

    def event_loop(self):

        while True:
            data = self.socket.recv_pyobj()
            print "Worker %s" % self.name
            print 'id = ', self.id_number
            print 'data=',data
            if data == 'quit':
                break

            self.socket.send_pyobj(20.0)

