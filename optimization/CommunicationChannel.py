
import multiprocessing
from multiprocessing import Queue,Event
from Queue import Empty

class CommunicationChannel(object):
    def __init__(self):
        self.listening_queue = Queue()
        self.listening_queue_lock = multiprocessing.Lock()
        self.queue_timeout = 1

    def send(self,msg):

        self.listening_queue_lock.acquire()
        self.listening_queue.put(msg)
        self.listening_queue_lock.release()

    def send_easy(self,*msg_pieces):

        self.listening_queue_lock.acquire()
        self.listening_queue.put(msg_pieces)
        self.listening_queue_lock.release()


    def retrieve(self):
        # return self.listening_queue.get(True, self.queue_timeout)
        return self.listening_queue.get()


# def a_fcn(*args):
#     print 'this is =',args
#
# a_fcn(1,2,3)