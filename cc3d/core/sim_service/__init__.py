# Access for customizing a service proxy interface on the fly by a service
from .ServiceFunctionRegistry import service_function

# Implementation service generators
from cc3d.core.sim_service.impl.CC3D.PyServiceCC3D import service_cc3d
from cc3d.core.sim_service.impl.RoadRunner.PyServiceRoadRunner import service_roadrunner


def close():
    """
    Closes module managers
    :return: None
    """
    from .managers import close_services
    close_services()
