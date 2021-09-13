import sys, os
try:
    import numpy
except ImportError :
    print(f'version {sys.version}')
    print(f'version_info {sys.version_info}')
    print(f'python install path ', os.__file__)
    print ("Could not find numpy include dir please make sure numpy is installed")

print (numpy.get_include())
