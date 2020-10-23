from cc3d.core.sim_service.managers import ServiceManagerLocal
from cc3d.core.sim_service.service_factory import process_factory
from cc3d.core.sim_service.service_wraps import TypeProcessWrap

from cc3d.core.sim_service.impl.CC3D.CC3DSimService import CC3DSimService

SERVICE_NAME = "CC3DSimService"


class CC3DSimServiceWrap(TypeProcessWrap):
    _process_cls = CC3DSimService


ServiceManagerLocal.register_service(SERVICE_NAME, CC3DSimServiceWrap)


def service_cc3d(*args, **kwargs):
    return process_factory(SERVICE_NAME, *args, **kwargs)
