from multiprocessing import Pipe, Process

from .basics import standard_process_name
from .messages import safe_transmit, worker_transmit


class ProcessContainer(Process):
    """
    Basic container for a service process
    An instance of this class instantiates a service environment spawns an underlying service process
    """
    def __init__(self, _conn,
                 _process_cls,
                 *args, **kwargs):
        """
        :param _conn: {multiprocessing.connection.Connection} X-process connection, process side
        :param _process_cls: process-launching class
        :param args: process-launching positional arguments
        :param kwargs: process-launching keyword arguments
        """
        super().__init__()
        self._conn = _conn
        self._proc = _process_cls(*args, **kwargs)

    def _instantiate_service_environment(self):
        # Initialize service function registry for this process
        process_name = standard_process_name()
        from .ServiceFunctionRegistry import ServiceFunctionRegistry
        ServiceFunctionRegistry.process_name = process_name
        _sfunc_receiver_conn, ServiceFunctionRegistry.receiver_conn = Pipe(False)
        ServiceFunctionRegistry.parent = self

        # Confirm launch by returning process id and connection to service function registry
        self._conn.send((process_name, _sfunc_receiver_conn))

    def run(self):
        self._instantiate_service_environment()

        # Communication protocol: container side
        safe_transmit(self._conn)(worker_transmit, self._conn, self._proc)
