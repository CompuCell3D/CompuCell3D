"""
CC3D core implementation of simservice TypeProcessWrap and registration as a simulation service
"""
from simservice.managers import ServiceManagerLocal
from simservice.service_factory import process_factory
from simservice.service_wraps import TypeProcessWrap

from cc3d.core.simservice.CC3DSimService import CC3DSimService

SERVICE_NAME = "CC3DSimService"


class CC3DSimServiceWrap(TypeProcessWrap):
    _process_cls = CC3DSimService


ServiceManagerLocal.register_service(SERVICE_NAME, CC3DSimServiceWrap)


def service_cc3d(*args, **kwargs):
    return process_factory(SERVICE_NAME, *args, **kwargs)
