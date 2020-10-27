from multiprocessing.connection import Pipe
from threading import Thread


class ConnectionMessage:
    """
    Standard message class for function calling between processes via a Pipe
    """

    TERMINATOR = "ConnectionMessageTerminator"

    def __init__(self, _command, *args, **kwargs):
        """
        :param _command: command name
        :param args: {tuple} command positional arguments
        :param kwargs: {dict} command keyword arguments
        """
        self.command = _command
        self.args = args
        self.kwargs = kwargs

    def __call__(self, _functor):
        """
        :param _functor: object with attribute *self.command*
        :return: evaluation return of *self.command* on *_functor* with stored arguments
        """
        if self.command is None:
            return _functor(*self.args, **self.kwargs)
        return getattr(_functor, self.command)(*self.args, **self.kwargs)

    @staticmethod
    def terminator():
        """
        :return: standard message to predicate connection termination
        """
        return ConnectionMessage(ConnectionMessage.TERMINATOR)

    @property
    def is_terminator(self) -> bool:
        """
        :return: {bool} True if this is a terminator message
        """
        return self.command == self.TERMINATOR


connection_terminator = ConnectionMessage(ConnectionMessage.TERMINATOR)


def dispatch_transmit(conn, _cmd, *args, **kwargs):
    """
    Generic communication protocol: dispatcher side
    :param conn: {multiprocessing.connection.Connection} dispatcher-worker connection, dispatcher side
    :param _cmd: function name
    :param args: {tuple} function positional arguments
    :param kwargs: {dict} function keyword arguments
    :return: function return value
    """
    msg = ConnectionMessage(_cmd, *args, **kwargs)
    conn.send(msg)
    val = conn.recv()
    return val


def dispatch_terminate(conn):
    """
    Generic communication protocol: terminate worker side
    :param conn: {multiprocessing.connection.Connection} dispatcher-worker connection, dispatcher side
    :return: None
    """
    conn.send(connection_terminator)
    conn.recv()


def worker_transmit(conn, func):
    """
    Generic communication protocol: worker side
    Blocks until routine is terminated by dispatcher
    :param conn: {multiprocessing.connection.Connection} dispatcher-worker connection, worker side
    :param func: functor with supporting evaluation by ConnectionMessage
    :return: None
    """
    while True:
        msg: ConnectionMessage = conn.recv()
        if msg.is_terminator:
            # Send acknowledgment of termination before returning
            conn.send(None)
            return
        else:
            # Call functor with arguments in message
            val = msg(func)
            conn.send(val)


def safe_transmit(conn=None, debug: bool = False):
    """
    Wrap to do communication protocol with safe handling of common exceptions
    :param conn: connection
    :param debug: {bool} option for notifications when pipes break or close
    :return:
    """
    def wrapper(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BrokenPipeError:
            if debug and conn is not None:
                print(f"Pipe has been broken: {conn}")
            return None
        except EOFError:
            if debug and conn is not None:
                print(f"Pipe has been closed: {conn}")
            return None
    return wrapper


class RemoteFunctionEvaluator:
    """
    Safe way to evaluate a function in a different process
    """
    def __init__(self, conn, func=None):
        """
        :param conn: {multiprocessing.connection.Connection} RemoteFunctionEvaluator-RemoteFunctionWorker connection,
        RemoteFunctionEvaluator side
        :param func: function to evaluate
        """
        self.__conn = conn
        if func is not None:
            self.__name__ = func.__name__
        else:
            self.__name__ = f"RemoteFunctionEvaluator_{conn.__name__}"

    def __call__(self, *args, **kwargs):
        return safe_transmit(self.__conn)(dispatch_transmit, self.__conn, None, *args, **kwargs)


class ServiceFunctionConnectionMessage:
    """
    Standard message class for service function callbacks with a RemoteFunctionEvaluator - RemoteFunctionWorker pair
    """
    def __init__(self, service_name: str, function_name: str, evaluator: RemoteFunctionEvaluator):
        """
        :param service_name: {str} service name
        :param function_name: {str} function name
        :param evaluator: {RemoteFunctionEvaluator} evaluator
        """
        self.service_name = service_name
        self.function_name = function_name
        self.evaluator = evaluator

    def __call__(self):
        return self.service_name, self.function_name, self.evaluator


class RemoteFunctionWorker(Thread):
    """
    Save way to send function evaluations to a different process

    DEV NOTE: this is safer than deriving from multiprocessing.Process, since potential applications include
    instantiation during bootstrap phase of parent processes
    """
    def __init__(self, conn, func, daemon: bool = False):
        """
        :param conn: {multiprocessing.connection.Connection} RemoteFunctionEvaluator-RemoteFunctionWorker connection,
        RemoteFunctionWorker side
        :param func: function to evaluate
        """
        super().__init__(daemon=daemon)
        self.__conn = conn
        self.__func = func

    def run(self) -> None:
        safe_transmit(self.__conn)(worker_transmit, self.__conn, self.__func)


def remote_function_factory(_func, daemon: bool = False) -> (RemoteFunctionEvaluator, RemoteFunctionWorker):
    """
    Generate a pipe to evaluate a function in a different process
    Should be generated in the process that defines the function, from which the evaluator can be piped elsewhere
    Safe so long as the function and its returned data can be serialized
    :param _func: function
    :param daemon: {bool} daemon argument to RemoteFunctionWorker instance
    :return: {RemoteFunctionEvaluator, RemoteFunctionWorker} evaluator-worker pair
    """
    e_conn, w_conn = Pipe()
    evaluator = RemoteFunctionEvaluator(e_conn, _func)
    worker = RemoteFunctionWorker(w_conn, _func, daemon=daemon)
    worker.start()
    return evaluator, worker
