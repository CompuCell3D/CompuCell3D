"""
Support for running CC3D core in a simservice environment
"""

# Check for simservice installation
try:
    import simservice
except ModuleNotFoundError as e:
    print('simservice could not be imported. Check your environment and install simservice if necessary')
    raise e

# Implementation service generator
from .PyServiceCC3D import service_cc3d

# Forwarding service function interface
service_function = simservice.service_function
