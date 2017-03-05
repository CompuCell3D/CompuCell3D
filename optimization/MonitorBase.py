from CommunicationChannel import *
# from messaging import dbgMsg
import command_enums as ce

from collections import namedtuple

Option = namedtuple('Option', ['name', 'default_value'])

class OptionsObject(object):
    def __init__(self):
        pass

    def OP(self, name, default_value=None):
        return Option(name=name, default_value=default_value)

    def init_options(self, option_list, options={}):
        for option in option_list:
            if hasattr(self, option.name): continue
            try:
                setattr(self, option.name, options[option.name])
                # print 'option_name=', option.name, ' val=', options[option.name], \
                #     ' value_check = ', getattr(self, option.name)
            except LookupError:
                setattr(self, option.name, option.default_value)

class MonitorBase(OptionsObject):

    def __init__(self,**options):
        OP = self.OP
        option_list = [
            OP(name='id_number',default_value=-1),
            OP(name='name',default_value='generic_monitor'),
            OP(name='configuration'),
            OP(name='session_data'),
            ]


        self.init_options(option_list, options)

        self.communication_channel = CommunicationChannel()

        # router related members
        self.router_channel = None
        self.id_name_registry = None

        # callback dictionary
        self.command_dispatching_dict = {}

        self.shared_buffer = None
        self.sample_len = 0 # this belongs elsewhere - SHARED_BUFFER

        # shared buffers
        self.shared_buffer_specs = {}
        self.shared_buffer_inventory = {}


    def get_addressee_id(self,name):
        try:
            return self.id_name_registry.get_id(name)
        except KeyError:
            raise RuntimeError('COuld not find monitor '+name+' in the registry.')


    def get_shared_buffer_specs(self):
        for name, length in self.shared_buffer_specs.items():
            yield name, length

    def set_shared_buffer(self,name, buf):
        self.shared_buffer_inventory[name] = buf

    def initialize(self):
        """
        Function called inside 'run' function before monitor enters event loop.
        It provides a way to do deferred initialization. Usually such function sends
        notification about completion of the initialization.
        :return: None
        """
        pass

    def finish(self):
        """
        Function called inside 'run' function after monitor exits event loop.
        It provides a way to do cleanup after event loop. Usually such function performs
        various cleanup tasks
        :return: None
        """
        pass

    def set_id_name_registry(self,id_name_registry):
        self.id_name_registry = id_name_registry


    def set_router_channel(self,router_channel):
        self.router_channel = router_channel

    def set_sample_len(self, sample_len):
        self.sample_len = sample_len

    def get_sample_len(self):
        return self.sample_len

    def stop(self):

        self.communication_channel.send([ce.STOP_PROCESS])

    def get_communication_channel(self):
        return self.communication_channel

    def get_id(self):
        return self.id_number

    def get_name(self):
        return self.name

    # def set_shared_buffer(self,shared_buffer):
    #     self.shared_buffer = shared_buffer
    #
    # def get_shared_buffer_sync(self):
    #     from common.SharedBuffer import SharedBuffersAccessor
    #
    #     return SharedBuffersAccessor(self.shared_buffer)
    #
    # def get_shared_buffer(self):
    #     return self.shared_buffer


    def get_shared_buffer_sync(self, name):
        from common.SharedBuffer import SharedBuffersAccessor

        return SharedBuffersAccessor(self.shared_buffer_inventory[name])


    def process_message(self, msg):

        command_id = msg[ce.MESSAGE_CONTENT_START_FIELD]

        try:
            command_handler = self.command_dispatching_dict[command_id]
        except:
            return

        command_handler(msg)

    def hard_stop(self):
        raise StopIteration('MonitorProcess: '+str(self.get_id())+' stop exception')

    def event_loop(self):

        while True:

            try:
                msg = self.communication_channel.retrieve()

                if msg[ce.FROM_FIELD] == ce.STOP_PROCESS: break

                self.process_message(msg=msg)

            except StopIteration, e:
                print 'GOT STOP ITERATION REQUEST ', str(e)
                # dbgMsg('GOT STOP ITERATION REQUEST ', str(e))
                break

    def run(self):

        self.initialize()
        self.event_loop()
        self.finish()
