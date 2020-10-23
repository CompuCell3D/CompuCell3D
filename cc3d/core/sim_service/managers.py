from multiprocessing.managers import BaseManager


class _ServiceManagerLocal(BaseManager):
    pass


class ServiceManagerLocal:

    manager = None
    started = False

    service_registry = dict()
    function_registry = dict()

    @staticmethod
    def _on_demand_manager():
        if ServiceManagerLocal.manager is None:
            ServiceManagerLocal.manager = _ServiceManagerLocal()
            print(f"Initialized service manager {ServiceManagerLocal.manager}")

    @classmethod
    def start(cls, *args, **kwargs):
        cls._on_demand_manager()
        if not cls.started:
            cls.manager.start(*args, **kwargs)
            cls.started = True

    @classmethod
    def _register_callable(cls, _name, _callable):
        cls.manager.register(_name, _callable)
        manager = cls.manager
        impl = getattr(manager, _name)
        setattr(cls, _name, impl)

    @classmethod
    def register_service(cls, _service_name, _service_wrap):
        cls._on_demand_manager()

        if _service_name in cls.service_registry.keys():
            raise AttributeError(f"Service with name {_service_name} has already been registered")

        cls.service_registry[_service_name] = _service_wrap
        cls._register_callable(_service_name, _service_wrap)

    @classmethod
    def register_function(cls, _function_name, _func):
        cls._on_demand_manager()

        if _function_name in cls.function_registry.keys():
            raise AttributeError(f"Function with name {_function_name} has already been registered")

        cls.function_registry[_function_name] = _func
        cls._register_callable(_function_name, _func)

    @classmethod
    def shutdown(cls):
        if cls.started:
            cls.manager.shutdown()
            cls.manager = None
            cls.started = False

    @classmethod
    def is_registered(cls, _service_name):
        return _service_name in cls.service_registry.keys()


class ServiceFunctionManager(BaseManager):
    pass


def close_services():
    ServiceManagerLocal.shutdown()
