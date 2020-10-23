from .messages import ServiceFunctionConnectionMessage, remote_function_factory


class NotAServiceError(Exception):
    """Raise when attempts are made to register a service function outside of a sim_service environment"""
    pass


class ServiceFunctionRegistry:
    """
    Registers service function on service process side and relays them to receiver in main process
    """

    # Connection to main process service function receiver; set by parent process during environment setup
    receiver_conn = None
    # Name of this process; set by parent process during environment setup
    process_name = None
    # Service process; set by parent process during environment setup
    # If this isn't set, then this class is being improperly accessed
    parent = None

    # Worker storage by function name
    # These calculate service functions on demand by the client side and send them back with threading
    workers = dict()

    @classmethod
    def register_function(cls, func, function_name: str = None):
        if cls.receiver_conn is None:
            raise NotAServiceError

        if function_name is None:
            function_name = func.__name__

        if function_name in cls.workers.keys():
            raise ValueError(f"Function with name {function_name} has already been registered as a service function")

        evaluator, worker = remote_function_factory(func)
        cls.workers[function_name] = (func, worker)
        msg = ServiceFunctionConnectionMessage(cls.process_name, function_name, evaluator)
        cls.receiver_conn.send(msg)


def service_function(func, function_name: str = None):
    """
    Service function registrator: makes ad-hoc modifications to the client-side interface
    This can be used within service specification to make an internal method available as a method on a
    client-side service proxy

    E.g., if MyServiceWrap defines a service wrap, the underlying service of which makes the following specification

    def f():
        print("Hello from the server side!")

    Then f() can be made available on a proxy of the service on the client side by declaring inside the service,

    service_function(f)

    Then on the client side, the following can then be performed:

    my_service = MyServiceWrap()
    my_service.run()
    my_service.f()

    Note that the availability of a service function depends on the specification of the underlying service. At various
    stages of usage, a service may or may not have yet made a service function available.

    :param func: function to make available
    :param function_name: {str} optional name assignment; if not specified, then the proxy method will be named the
    same as the registered function
    Useful for attaching to methods of the same name to the same service as a service function
    :return: None
    """

    try:
        ServiceFunctionRegistry.register_function(func, function_name)
    except NotAServiceError:
        print(f"Service functions can only be registered in a service environment ({func}, {function_name})")
