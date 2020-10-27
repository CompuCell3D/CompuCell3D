from cc3d.core.sim_service.managers import ServiceManagerLocal
from cc3d.core.sim_service.service_factory import process_factory
from cc3d.core.sim_service.service_wraps import TypeProcessWrap

from .RoadRunnerSimService import RoadRunnerSimService

SERVICE_NAME = "RoadRunnerSimService"


class RoadRunnerSimServiceWrap(TypeProcessWrap):
    _process_cls = RoadRunnerSimService


ServiceManagerLocal.register_service(SERVICE_NAME, RoadRunnerSimServiceWrap)


def service_roadrunner(*args, **kwargs):
    return process_factory(SERVICE_NAME, *args, **kwargs)

