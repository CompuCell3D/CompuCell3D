from threading import Thread

from .messages import safe_transmit


class _ServiceFunctionContainer:
    """
    Container for holding service functions by service proxy
    """
    def __init__(self):
        # Service function storage
        self._sfuncs = dict()

    def _register_function(self, function_name: str, evaluator):
        """
        Register service function
        :param function_name: {str} name of function
        :param evaluator: {RemoteFunctionEvaluator} remote function evaluator
        :return: None
        """
        self._sfuncs[function_name] = evaluator
        setattr(self, function_name, evaluator)


class _ServiceFunctionConnectionWorker(Thread):
    """
    An instance watches a registry-receiver connection, notifies the receiver that there's work to do and adds a
    newly registered service function to a service proxy instance
    """
    def __init__(self, _service_name, _sfunc_reg_conn, _conn_flusher, _sfunc_container, _attr_adder):
        super().__init__(daemon=True)
        self._sfunc_reg_conn = _sfunc_reg_conn
        self._conn_flusher = _conn_flusher
        self._sfunc_container = _sfunc_container
        self._attr_adder = _attr_adder

        self._added_functions = list()

    def run(self):
        while safe_transmit()(self._check_connection) is not None:
            continue

    def _check_connection(self):
        if self._sfunc_reg_conn.closed:
            return None
        if self._sfunc_reg_conn.poll():
            self._conn_flusher()
            for f_name, evaluator in self._sfunc_container._sfuncs.items():
                if f_name not in self._added_functions:
                    self._added_functions.append(f_name)
                    self._attr_adder(f_name, evaluator)
        return True


class ServiceFunctionReceiver:
    """
    Handles callbacks for instantiating service functions of service proxies

    Service process sends registered service functions via Pipe
    Processing generates service function containers for accessing underlying service function on the client side
    Processing is invoked by workers that monitors pipes between registry in the service process and the receiver in
    the main process
    """

    KEY_CONN = "Connection"
    KEY_CONT = "Container"

    # Service container storage
    service_containers = dict()
    # Service function connection worker storage
    workers = dict()

    @classmethod
    def register_service(cls, service_proxy, service_conn):
        """
        Register service with receiver

        A service process should be registered before being deployed, since the receiver must be ready to handle
        service function callbacks during service process activity
        :param service_proxy: service proxy instance
        :param service_conn: {multiprocessing.connection.Connection} service function callback connection; receiver side
        :return: None
        """
        service_name = service_proxy.process_name()
        if service_name in cls.service_containers.keys():
            raise AttributeError(f"Service {service_name} has already been registered")
        cls.service_containers[service_name] = {cls.KEY_CONN: service_conn,
                                                cls.KEY_CONT: _ServiceFunctionContainer()}

        # Spawn a worker to handle service function registration during internal service routines
        service_name = service_proxy.process_name()
        if service_name not in cls.service_containers.keys():
            raise AttributeError(f"Service {service_name} has not been registered")

        _flush_connection = cls._flush_connection
        __setattr__ = service_proxy.__setattr__

        def _flusher():
            _flush_connection(service_name)

        def _attr_adder(name, val):
            __setattr__(name, val)

        worker = _ServiceFunctionConnectionWorker(service_name,
                                                  cls.service_containers[service_name][cls.KEY_CONN],
                                                  _flusher,
                                                  cls.service_containers[service_name][cls.KEY_CONT],
                                                  _attr_adder)
        worker.start()
        cls.workers[service_name] = worker

    @classmethod
    def _flush_connection(cls, service_name: str):
        """
        Process all pending service function registrations in the Pipe of a service process
        :param service_name: {str} name of service process
        :return: None
        """
        conn = cls.service_containers[service_name][cls.KEY_CONN]

        while conn.poll():
            msg = conn.recv()  # ServiceFunctionConnectionMessage
            process_name, function_name, evaluator = msg()
            if process_name != service_name:
                print(f"Incorrect pipe usage {process_name} -> {service_name}. Rejecting")
                continue
            cls.service_containers[service_name][cls.KEY_CONT]._register_function(function_name, evaluator)
