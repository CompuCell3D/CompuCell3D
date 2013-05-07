try:
    import numpy
except ImportError :
    print "Could not find numpy include dir please make sure numpy is installed"

print numpy.get_include()
